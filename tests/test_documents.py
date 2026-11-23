import io

import pymupdf
import pytest
from PIL import Image

from agent.documents import DocumentError, load_document, sniff
from tests.conftest import SAMPLES, make_settings


def pdf(pages=1, text="word " * 40, **save):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 500, 700), text)
    return bytes(doc.tobytes(**save))


def png(size=(300, 400), mode="RGB", color="white"):
    out = io.BytesIO()
    Image.new(mode, size, color).save(out, "PNG")
    return out.getvalue()


def test_sniff_uses_signatures_not_names():
    assert sniff(pdf()) == "pdf"
    assert sniff(png()) == "png"
    assert sniff((SAMPLES / "photo.jpg").read_bytes()) == "jpeg"
    assert sniff(b"<html>%PDF") is None
    assert sniff(b"") is None
    assert sniff(b"MZ\x90\x00") is None


def test_text_layer_is_preferred():
    d = load_document(pdf(), make_settings())
    assert d.kind == "pdf_text" and d.images == [] and "word" in d.text


def test_one_page_without_text_means_the_whole_pdf_goes_to_vision():
    doc = pymupdf.open(stream=pdf(), filetype="pdf")
    doc.new_page()
    d = load_document(bytes(doc.tobytes()), make_settings())
    assert d.kind == "pdf_scan" and len(d.images) == 2 and d.text is None


def test_images_are_bounded_and_rotated_upright():
    s = make_settings(vision_max_edge=800)
    out = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 clockwise to display
    Image.new("RGB", (3000, 1000), "white").save(out, "JPEG", exif=exif)
    image = Image.open(io.BytesIO(load_document(out.getvalue(), s).images[0]))
    assert image.size[1] == 800 and image.size[0] < image.size[1]


def test_transparent_png_is_flattened_on_white():
    image = Image.open(
        io.BytesIO(load_document(png(mode="RGBA", color=(0, 0, 0, 0)), make_settings()).images[0])
    )
    assert image.getpixel((5, 5))[0] > 250


def test_huge_page_is_rendered_small():
    doc = pymupdf.open()
    doc.new_page(width=14400, height=14400)
    d = load_document(bytes(doc.tobytes()), make_settings(vision_max_edge=500))
    assert max(Image.open(io.BytesIO(d.images[0])).size) <= 501


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"hello", "Only PDF"),
        (b"%PDF-1.7 garbage", "PDF"),
        (b"\x89PNG\r\n\x1a\n" + b"junk" * 20, "could not be opened"),
        (b"\xff\xd8\xff" + b"junk" * 20, "could not be opened"),
        (pdf(6), "6 pages"),
        (pdf(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y"), "password"),
    ],
)
def test_bad_files_give_a_readable_error(data, message):
    with pytest.raises(DocumentError, match=message):
        load_document(data, make_settings())


def test_decompression_bomb_is_refused(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(DocumentError, match="too large"):
        load_document(png((100, 100)), make_settings())
