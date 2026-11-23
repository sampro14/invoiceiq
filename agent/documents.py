"""Reading an upload: signature check, text layer, page images. Everything stays in memory."""

import io
import threading
from dataclasses import dataclass, field

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

from agent.config import Settings
from agent.state import FileKind

pymupdf.TOOLS.mupdf_display_errors(False)
Image.MAX_IMAGE_PIXELS = 50_000_000

# MuPDF is not thread-safe, and documents are read on worker threads.
_MUPDF = threading.Lock()


class DocumentError(Exception):
    """The file cannot be processed. The message is shown to the user."""


@dataclass
class Document:
    kind: FileKind
    page_count: int
    text: str | None = None
    images: list[bytes] = field(default_factory=list)
    # Written by the reader only when the second model cannot see images.
    transcript: str | None = None


class DocumentStore:
    """Uploads and their decoded pages, by run. Never checkpointed, dropped when the run pauses."""

    def __init__(self) -> None:
        self._raw: dict[str, bytes] = {}
        self._docs: dict[str, Document] = {}

    def put(self, run_id: str, data: bytes) -> None:
        self._raw[run_id] = data

    def raw(self, run_id: str) -> bytes:
        return self._raw[run_id]

    def set_document(self, run_id: str, document: Document) -> None:
        self._docs[run_id] = document
        self._raw.pop(run_id, None)

    def document(self, run_id: str) -> Document:
        return self._docs[run_id]

    def drop(self, run_id: str) -> None:
        self._raw.pop(run_id, None)
        self._docs.pop(run_id, None)

    def __len__(self) -> int:
        return len(self._raw.keys() | self._docs.keys())


def sniff(data: bytes) -> str | None:
    """File type by signature; extensions and content types are the client's claim."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if b"%PDF-" in data[:1024]:
        return "pdf"
    return None


def load_document(data: bytes, settings: Settings) -> Document:
    kind = sniff(data)
    if kind is None:
        raise DocumentError("Only PDF, PNG and JPG files are supported.")
    if kind != "pdf":
        return _load_image(data, settings)
    with _MUPDF:
        return _load_pdf(data, settings)


def _load_pdf(data: bytes, settings: Settings) -> Document:
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise DocumentError("This PDF could not be opened.") from e
    with pdf:
        if pdf.needs_pass:
            raise DocumentError("This PDF is password protected.")
        if pdf.page_count == 0:
            raise DocumentError("This PDF has no pages.")
        if pdf.page_count > settings.max_pages:
            raise DocumentError(
                f"This PDF has {pdf.page_count} pages; the limit is {settings.max_pages}."
            )
        try:
            pages = [str(page.get_text("text", sort=True)) for page in pdf.pages()]
            if all(len("".join(p.split())) > settings.min_chars_per_page for p in pages):
                return Document("pdf_text", pdf.page_count, text="\n\f\n".join(pages))
            images = [_render(page, settings) for page in pdf.pages()]
        except DocumentError:
            raise
        except Exception as e:
            raise DocumentError("This PDF could not be read.") from e
        return Document("pdf_scan", pdf.page_count, images=images)


def _render(page: pymupdf.Page, settings: Settings) -> bytes:
    longest = max(page.rect.width, page.rect.height)
    if longest <= 0:
        raise DocumentError("This PDF has an empty page.")
    # Zoom is derived from the target size, so a page with a huge media box cannot blow up memory.
    zoom = settings.vision_max_edge / longest
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return bytes(pixmap.tobytes("jpeg", jpg_quality=settings.vision_jpeg_quality))


def _load_image(data: bytes, settings: Settings) -> Document:
    try:
        with Image.open(io.BytesIO(data), formats=("PNG", "JPEG")) as opened:
            opened.draft("RGB", (settings.vision_max_edge * 2, settings.vision_max_edge * 2))
            image = ImageOps.exif_transpose(opened)
            if image.mode in ("RGBA", "LA", "P"):
                # Flatten onto white: transparent scans would otherwise turn black.
                rgba = image.convert("RGBA")
                flat = Image.new("RGB", rgba.size, "white")
                flat.paste(rgba, mask=rgba.getchannel("A"))
                image = flat
            image = image.convert("RGB")
            image.thumbnail(
                (settings.vision_max_edge, settings.vision_max_edge), Image.Resampling.LANCZOS
            )
            out = io.BytesIO()
            image.save(out, "JPEG", quality=settings.vision_jpeg_quality)
    except Image.DecompressionBombError as e:
        raise DocumentError("This image is too large to process.") from e
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise DocumentError("This image could not be opened.") from e
    return Document("image", 1, images=[out.getvalue()])
