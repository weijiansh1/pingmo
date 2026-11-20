"""Reconstruct native scanned PDF pages from their positioned image strips.

No OCR or image enhancement is used. Page numbers are PDF positions (1-based).
This file is specific to this supplied PDF; unsupported transforms are rejected.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "GJB_2874-1997_电传操纵系统飞机的飞行品质.pdf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pages", nargs="+", type=int)
    args = parser.parse_args()
    reader = PdfReader(PDF)
    out = HERE / "pages"
    out.mkdir(exist_ok=True)
    for number in args.pages:
        page = reader.pages[number - 1]
        width, height = map(lambda v: int(round(float(v))), (page.mediabox.width, page.mediabox.height))
        canvas = Image.new("RGB", (width, height), "white")
        matrix = None
        for values, operator in page.get_contents().operations:
            if operator == b"cm":
                matrix = list(map(float, values))
            elif operator == b"Do":
                a, b, c, d, e, f = matrix
                if b != 0 or c != 0 or a <= 0 or d <= 0:
                    raise ValueError(f"Unsupported image transform on page {number}: {matrix}")
                strip = page.images[str(values[0])].image.convert("RGB")
                if strip.size != (round(a), round(d)):
                    raise ValueError(f"Non-native scale on page {number}")
                canvas.paste(strip, (round(e), round(height - f - d)))
        path = out / f"pdf_{number:03d}.png"
        canvas.save(path)
        print(f"PDF {number}: {width} x {height} -> {path.name}")
    manifest = {
        "source_pdf": str(PDF),
        "sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "pdf_pages": len(reader.pages),
        "text_layer": False,
        "method": "native_image_strips_positioned_using_pdf_cm_and_Do_no_OCR",
        "page_images": [{"pdf_page": int(p.stem.split("_")[1]), "path": str(p.relative_to(HERE)),
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(out.glob("pdf_*.png"))],
    }
    (HERE / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
