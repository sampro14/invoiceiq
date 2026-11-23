"""Synthetic invoices with exact ground truth.

Everything derives from a seed. Documents are locked by a hash of their *spec* (labels, layout,
perturbation) rather than of the rendered bytes, because PDF and JPEG bytes vary with the zlib and
libjpeg builds on each machine.
"""

import hashlib
import io
import json
import random
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

import pymupdf
from faker import Faker
from PIL import Image, ImageEnhance, ImageFilter
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

GENERATOR_VERSION = 1
CENT = Decimal("0.01")

GOODS = [
    ("Stainless steel hex bolts M12 x 60", "7318"), ("Hydraulic hose assembly 2 m", "4009"),
    ("Industrial safety gloves, pair", "4015"), ("Bearing 6204-2RS", "8482"),
    ("Corrugated carton 5-ply 450 x 300", "4819"), ("Lithium grease 500 g", "3403"),
    ("LED high-bay light 150 W", "9405"), ("Copper cable 4 sq mm, 90 m coil", "8544"),
    ("Annual maintenance, air compressor", "9987"), ("Site installation labour, per hour", "9954"),
    ("Pressure gauge 0-10 bar", "9026"), ("Epoxy floor paint 20 L", "3208"),
    ("Thermal transfer ribbon 110 mm", "9612"), ("Barcode labels 100 x 50, roll", "4821"),
    ("Pallet wrap film 23 micron", "3920"), ("Calibration service, weighing scale", "9983"),
    ("V-belt B-68", "4010"), ("Solenoid valve 24 V DC", "8481"), ("Cutting disc 105 mm", "6804"),
    ("Printer toner cartridge, black", "8443"),
]  # fmt: skip
STATES = [("27", "Maharashtra", "Pune"), ("29", "Karnataka", "Bengaluru"), ("24", "Gujarat", "Surat"),
          ("33", "Tamil Nadu", "Chennai"), ("07", "Delhi", "New Delhi"), ("36", "Telangana", "Hyderabad")]  # fmt: skip
VENDOR_WORDS = ["Sahyadri", "Deccan", "Konark", "Vindhya", "Malabar", "Triveni", "Nilgiri", "Aravalli",
                "Godavari", "Chetak", "Indus", "Saptagiri"]  # fmt: skip
VENDOR_TAILS = ["Engineering Pvt Ltd", "Industrial Supplies", "Trading Co", "Fasteners LLP", "Automation Pvt Ltd",
                "Packaging Works", "Electricals"]  # fmt: skip
UAE_VENDORS = [
    "Gulf Crescent Trading LLC",
    "Al Noor Industrial Supplies LLC",
    "Palm Bridge Facilities FZE",
]

Layout = Literal["ruled", "banner", "boxed", "plain"]
LAYOUTS: list[Layout] = ["ruled", "banner", "boxed", "plain"]


def q2(value: Decimal) -> Decimal:
    return value.quantize(CENT, ROUND_HALF_UP)


@dataclass
class Line:
    description: str
    hsn: str
    quantity: Decimal
    unit_price: Decimal
    line_total: Decimal


@dataclass
class Spec:
    seed: int
    country: Literal["IN", "AE"]
    layout: Layout
    vendor_name: str
    vendor_address: str
    tax_id: str
    buyer: str
    buyer_address: str
    number: str
    issue_date: date
    due_date: date
    po_number: str | None
    lines: list[Line]
    subtotal: Decimal
    discount: Decimal
    taxes: list[tuple[str, Decimal, Decimal]]  # label, rate, amount
    total: Decimal
    date_format: str = "%d-%m-%Y"
    # What gets printed, when it differs from the truth of the transaction.
    print_taxes: bool = True
    injections: list[dict[str, Any]] = field(default_factory=list)
    kind: Literal["pdf", "image"] = "pdf"
    degrade: dict[str, Any] = field(default_factory=dict)

    @property
    def currency(self) -> str:
        return "INR" if self.country == "IN" else "AED"

    def labels(self) -> dict[str, Any]:
        """Ground truth: what is printed on the page."""
        tax = sum((t[2] for t in self.taxes), Decimal(0)) if self.print_taxes else None
        rate = sum((t[1] for t in self.taxes), Decimal(0)) if self.print_taxes else None
        return {
            "document_type": "invoice",
            "vendor.name": self.vendor_name,
            "vendor.address": self.vendor_address,
            "vendor.tax_id": self.tax_id,
            "invoice.number": self.number,
            "invoice.issue_date": self.issue_date.isoformat(),
            "invoice.due_date": self.due_date.isoformat(),
            "invoice.currency": self.currency,
            "invoice.po_number": self.po_number,
            "amounts.subtotal": str(self.subtotal),
            "amounts.discount": str(self.discount) if self.discount else None,
            "amounts.tax_amount": None if tax is None else str(tax),
            "amounts.tax_rate": None if rate is None else str(rate),
            "amounts.total": str(self.total),
            "line_items": [
                {"description": ln.description, "quantity": str(ln.quantity),
                 "unit_price": str(ln.unit_price), "line_total": str(ln.line_total)}
                for ln in self.lines
            ],
        }  # fmt: skip

    def spec_hash(self) -> str:
        payload = {
            "generator": GENERATOR_VERSION, "seed": self.seed, "layout": self.layout, "labels": self.labels(),
            "injections": self.injections, "kind": self.kind, "degrade": self.degrade,
        }  # fmt: skip
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def render(self) -> bytes:
        pdf = render_pdf(self)
        return pdf if self.kind == "pdf" else degrade(pdf, self.seed, **self.degrade)


def gstin(rng: random.Random, state: str) -> str:
    """A well-formed GSTIN with a correct check character."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base = (
        state
        + "".join(rng.choices(letters, k=5))
        + "".join(rng.choices("0123456789", k=4))
        + rng.choice(letters)
        + rng.choice("123")
        + "Z"
    )
    alphabet = "0123456789" + letters
    total = 0
    for i, ch in enumerate(base):
        product = alphabet.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return base + alphabet[(36 - total % 36) % 36]


def make_spec(seed: int, country: Literal["IN", "AE"] = "IN", layout: Layout | None = None) -> Spec:
    rng = random.Random(f"invoiceiq-{seed}")
    fake = Faker("en_IN" if country == "IN" else "en_US")
    fake.seed_instance(seed)
    lines = []
    for description, hsn in rng.sample(GOODS, rng.randint(2, 7)):
        quantity = Decimal(rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 25, 40]))
        price = q2(Decimal(rng.randint(1500, 480000)) / 100)
        lines.append(Line(description, hsn, quantity, price, q2(quantity * price)))
    subtotal = sum((ln.line_total for ln in lines), Decimal(0))
    discount = q2(subtotal * Decimal("0.05")) if rng.random() < 0.2 else Decimal(0)
    taxable = subtotal - discount
    if country == "IN":
        code, state, city = rng.choice(STATES)
        vendor = f"{rng.choice(VENDOR_WORDS)} {rng.choice(VENDOR_TAILS)}"
        address = f"Plot {rng.randint(2, 220)}, {rng.choice(['MIDC', 'GIDC', 'Industrial Estate'])}, {city}, {state} {rng.randint(110001, 699999)}"
        tax_id = gstin(rng, code)
        rate = Decimal(rng.choice([5, 12, 18, 28]))
        if rng.random() < 0.6:
            half = rate / 2
            taxes = [(n, half, q2(taxable * half / 100)) for n in ("CGST", "SGST")]
        else:
            taxes = [("IGST", rate, q2(taxable * rate / 100))]
        buyer, buyer_address = (
            fake.company(),
            f"{rng.randint(1, 90)} Ring Road, {rng.choice(STATES)[2]}",
        )
        number = rng.choice([
            f"INV/{rng.randint(24, 26)}-{rng.randint(27, 29)}/{rng.randint(100, 9999):04d}",
            f"TI-{rng.randint(10000, 99999)}", f"GST-{rng.randint(1000, 9999)}",
        ])  # fmt: skip
    else:
        vendor = rng.choice(UAE_VENDORS)
        address = (
            f"Office {rng.randint(101, 2404)}, Al Quoz Industrial {rng.randint(1, 4)}, Dubai, UAE"
        )
        tax_id = "100" + "".join(rng.choices("0123456789", k=12))
        taxes = [("VAT", Decimal(5), q2(taxable * 5 / 100))]
        buyer, buyer_address = fake.company(), f"Warehouse {rng.randint(1, 40)}, Jebel Ali, Dubai"
        number = f"INV-{rng.randint(2025, 2026)}-{rng.randint(100, 9999):04d}"
    issued = date(2026, 1, 5) + timedelta(days=rng.randint(0, 290))
    return Spec(
        seed=seed, country=country, layout=layout if layout is not None else rng.choice(LAYOUTS),
        vendor_name=vendor, vendor_address=address, tax_id=tax_id,
        buyer=buyer, buyer_address=buyer_address, number=number,
        issue_date=issued, due_date=issued + timedelta(days=rng.choice([7, 15, 30, 45])),
        po_number=f"PO-{rng.randint(10000, 99999)}" if rng.random() < 0.6 else None,
        lines=lines, subtotal=subtotal, discount=discount, taxes=taxes,
        total=taxable + sum((t[2] for t in taxes), Decimal(0)),
        date_format=rng.choice(["%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%Y-%m-%d"]),
    )  # fmt: skip


def indian(value: Decimal) -> str:
    sign, text = ("-" if value < 0 else ""), f"{abs(value):.2f}"
    whole, frac = text.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups: list[str] = []
        while head:
            groups.insert(0, head[-2:])
            head = head[:-2]
        whole = ",".join([*groups, tail])
    return f"{sign}{whole}.{frac}"


def render_pdf(spec: Spec) -> bytes:
    fmt = indian if spec.country == "IN" else (lambda v: f"{v:,.2f}")
    buffer = io.BytesIO()
    c = Canvas(buffer, pagesize=A4, pageCompression=0, invariant=1)
    c.setTitle(f"Invoice {spec.number}")
    width, height = A4
    left, right = 42.0, width - 42.0
    ink, accent = HexColor("#1d2733"), {
        "ruled": HexColor("#1d2733"), "banner": HexColor("#1f4e5f"),
        "boxed": HexColor("#333333"), "plain": HexColor("#6b3e26"),
    }[spec.layout]  # fmt: skip
    tax_label = "GSTIN" if spec.country == "IN" else "TRN"
    y = height - 56

    def text(
        x: float,
        yy: float,
        s: str,
        size: float = 9,
        bold: bool = False,
        align: str = "l",
        color: Any = ink,
    ) -> None:
        c.setFillColor(color)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        {"l": c.drawString, "r": c.drawRightString, "c": c.drawCentredString}[align](x, yy, s)

    title = "TAX INVOICE"
    meta = [
        ("Invoice No", spec.number),
        ("Invoice Date", spec.issue_date.strftime(spec.date_format)),
        ("Due Date", spec.due_date.strftime(spec.date_format)),
    ]
    if spec.po_number:
        meta.append(("PO No", spec.po_number))

    if spec.layout == "banner":
        c.setFillColor(accent)
        c.rect(0, height - 92, width, 92, stroke=0, fill=1)
        text(left, height - 46, spec.vendor_name, 17, True, color=white)
        text(right, height - 46, title, 11, True, "r", white)
        text(left, height - 64, spec.vendor_address, 8.5, color=white)
        text(left, height - 78, f"{tax_label}: {spec.tax_id}", 8.5, color=white)
        y = height - 122
    elif spec.layout == "boxed":
        text(width / 2, y, spec.vendor_name, 15, True, "c")
        text(width / 2, y - 14, spec.vendor_address, 8.5, align="c")
        text(width / 2, y - 27, f"{tax_label}: {spec.tax_id}", 8.5, align="c")
        text(width / 2, y - 48, title, 11, True, "c")
        c.setStrokeColor(accent)
        c.rect(left - 8, 60, right - left + 16, height - 100, stroke=1, fill=0)
        y -= 76
    else:
        text(left, y, spec.vendor_name, 15, True, color=accent)
        text(right, y, title, 13, True, "r", accent)
        text(left, y - 15, spec.vendor_address, 8.5)
        text(left, y - 28, f"{tax_label}: {spec.tax_id}", 8.5)
        y -= 58

    text(left, y, "Bill To", 8.5, True, color=accent)
    text(left, y - 13, spec.buyer)
    text(left, y - 25, spec.buyer_address, 8.5)
    meta_x = right - 190
    for i, (k, v) in enumerate(meta):
        text(meta_x, y - i * 13, f"{k}:", 8.5, True)
        text(right, y - i * 13, v, 9, align="r")
    y -= 13 * max(len(meta), 3) + 22

    cols = {"sr": left, "desc": left + 24, "qty": right - 200, "price": right - 100, "total": right}
    if spec.layout in ("banner", "ruled"):
        c.setFillColor(accent)
        c.rect(left - 6, y - 5, right - left + 12, 18, stroke=0, fill=1)
        head_color = white
    else:
        c.setStrokeColor(accent)
        c.line(left - 6, y + 12, right + 6, y + 12)
        c.line(left - 6, y - 5, right + 6, y - 5)
        head_color = accent
    for key, label, align in (("sr", "#", "l"), ("desc", "Description", "l"), ("qty", "Qty", "r"),
                              ("price", f"Rate ({spec.currency})", "r"), ("total", f"Amount ({spec.currency})", "r")):  # fmt: skip
        text(cols[key], y, label, 8.5, True, align, head_color)
    y -= 20
    for i, ln in enumerate(spec.lines):
        if spec.layout == "banner" and i % 2:
            c.setFillColor(HexColor("#eef3f5"))
            c.rect(left - 6, y - 5, right - left + 12, 17, stroke=0, fill=1)
        text(cols["sr"], y, str(i + 1), 9)
        text(cols["desc"], y, ln.description, 9)
        text(cols["qty"], y, f"{ln.quantity:f}", 9, align="r")
        text(cols["price"], y, fmt(ln.unit_price), 9, align="r")
        text(cols["total"], y, fmt(ln.line_total), 9, align="r")
        y -= 17
    c.setStrokeColor(accent)
    c.line(left - 6, y + 8, right + 6, y + 8)
    y -= 8

    totals = [("Subtotal", fmt(spec.subtotal), False)]
    if spec.discount:
        totals.append(("Discount", fmt(-spec.discount), False))
    if spec.print_taxes:
        totals += [(f"{n} {r.normalize():f}%", fmt(a), False) for n, r, a in spec.taxes]
    totals.append(("Grand Total", f"{spec.currency} {fmt(spec.total)}", True))
    for label, value, bold in totals:
        text(right - 200, y, label, 10 if bold else 9, bold)
        text(right, y, value, 10 if bold else 9, bold, "r")
        y -= 16

    y -= 14
    text(
        left,
        y,
        "Payment by bank transfer within the due date. Goods once sold are not returned.",
        8,
    )
    for injection in spec.injections:
        y -= 16
        style = injection.get("style", "visible")
        color = white if style == "hidden" else HexColor("#555555")
        size = 1.5 if style == "tiny" else 8
        text(left, y, injection["text"], size, color=color)
    c.showPage()
    c.save()
    return buffer.getvalue()


def degrade(
    pdf: bytes,
    seed: int,
    *,
    dpi: int = 150,
    angle: float = 0.0,
    blur: float = 0.0,
    noise: int = 0,
    perspective: float = 0.0,
    shade: float = 0.0,
    contrast: float = 1.0,
    quality: int = 80,
) -> bytes:
    """Turn a clean PDF page into something like a scan or a phone photo."""
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    rng = random.Random(seed)
    if shade or noise:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if perspective:
        w, h = image.size
        d = perspective
        pad = Image.new("RGB", (int(w * 1.14), int(h * 1.1)), (92, 78, 66))
        # Each output corner samples a point slightly outside the page, so the page tapers.
        quad = (
            -w * 0.07 * (1 + d),
            -h * 0.05,
            -w * 0.05,
            h * 1.05,
            w * (1.06 + d / 3),
            h * 1.05,
            w * (1.05 + d),
            -h * 0.04,
        )
        page = image.convert("RGBA").transform(
            pad.size, Image.Transform.QUAD, quad, Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0)
        )
        pad.paste(page, (0, 0), page.getchannel("A"))
        image = pad
    if angle:
        image = image.rotate(
            angle,
            Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(228, 226, 220) if not perspective else (92, 78, 66),
        )
    if shade:
        # Rotate a larger gradient and crop, so no corner of the page falls outside it.
        ramp = Image.linear_gradient("L").resize((512, 512), Image.Resampling.BICUBIC)
        ramp = ramp.rotate(90 + rng.uniform(-30, 30), Image.Resampling.BICUBIC)
        gradient = ramp.crop((96, 96, 416, 416)).resize(image.size, Image.Resampling.BICUBIC)
        dark = ImageEnhance.Brightness(image).enhance(1 - shade)
        image = Image.composite(image, dark, gradient)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        grain = Image.frombytes(
            "L", image.size, random.Random(seed).randbytes(image.width * image.height)
        )
        grain = grain.point(lambda p: 128 + (p - 128) * noise // 128)
        image = Image.blend(image, Image.merge("RGB", (grain,) * 3), 0.12)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=quality)
    return out.getvalue()


def with_error(spec: Spec, error: str) -> tuple[Spec, str]:
    """Plant one error. Returns the spec and the rule that should catch it."""
    if error == "total":
        return replace(spec, total=spec.total + Decimal("450.00")), "subtotal_tax_total"
    if error == "subtotal":
        bump = Decimal("1000.00")
        return replace(
            spec, subtotal=spec.subtotal + bump, total=spec.total + bump
        ), "lines_to_subtotal"
    if error == "line":
        lines = [replace(ln) for ln in spec.lines]
        lines[0].line_total += Decimal("90.00")
        sub = spec.subtotal + Decimal("90.00")
        return replace(
            spec, lines=lines, subtotal=sub, total=spec.total + Decimal("90.00")
        ), "line_math"
    if error == "missing_tax":
        return replace(spec, print_taxes=False), "subtotal_tax_total"
    if error == "tax_id":
        return replace(spec, tax_id=spec.tax_id[:-2]), "tax_id"
    if error == "due_date":
        return replace(spec, due_date=spec.issue_date - timedelta(days=12)), "dates"
    raise ValueError(error)
