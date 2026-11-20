"""Point clouds of the parameter-space figure as small vector PDFs, standard library only.

TeX runs out of main memory with 5,000 marks per panel, so each panel's marks
are written here and placed by pgfplots with ``\\addplot graphics``; axes,
boundaries and text stay in LaTeX.  The page spans exactly the axis box, with
log axes mapped linearly in log space.

Usage: python3 -I make_scatter_pdf.py <data dir>
"""
import math
import sys
import zlib
from pathlib import Path

DATA = Path(sys.argv[1]).resolve()
CM = 72.0 / 2.54
WIDTH, HEIGHT = 3.85 * CM, 3.0 * CM          # matches width/height of the scale-only axes

# panel: x column, y column, x log, y log, x range, y range (must match paper.tex)
PANELS = {
    "a": ("tr", "wd", True, True, (0.15, 16), (0.18, 7)),
    "b": ("wd", "zd", True, False, (0.18, 7), (0.0, 0.75)),
    "c": ("nalpha", "wsp", True, True, (2, 70), (0.4, 15)),
    "d": ("zsp", "taut", False, False, (0.05, 1.3), (0.0, 0.33)),
    "e": ("psib", "dbk", False, True, (-360, 0), (0.3, 2000)),
    "f": ("wp", "zp", True, False, (0.004, 0.25), (-0.12, 0.6)),
}
# drawing order and mark style per level (colours of the validated reference palette)
LEVELS = [
    ("L1", "circle", (0x2A, 0x78, 0xD6), 0.55, 0.5),
    ("L2", "triangle", (0xEB, 0x68, 0x34), 0.60, 0.7),
    ("L3", "square", (0x1B, 0xAF, 0x7A), 0.60, 0.5),
    ("OOD", "cross", (0x52, 0x51, 0x4E), 0.70, 0.8),
]
K = 0.5523


def read(path):
    lines = path.read_text().splitlines()
    head = lines[0].split()
    return [dict(zip(head, line.split())) for line in lines[1:] if line.strip()]


def scale(v, lo, hi, log, size):
    if log:
        if v <= 0:
            return None
        return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * size
    return (v - lo) / (hi - lo) * size


def mark(shape, x, y, s):
    if shape == "circle":
        c = K * s
        return (f"{x + s:.2f} {y:.2f} m {x + s:.2f} {y + c:.2f} {x + c:.2f} {y + s:.2f} {x:.2f} {y + s:.2f} c "
                f"{x - c:.2f} {y + s:.2f} {x - s:.2f} {y + c:.2f} {x - s:.2f} {y:.2f} c "
                f"{x - s:.2f} {y - c:.2f} {x - c:.2f} {y - s:.2f} {x:.2f} {y - s:.2f} c "
                f"{x + c:.2f} {y - s:.2f} {x + s:.2f} {y - c:.2f} {x + s:.2f} {y:.2f} c f\n")
    if shape == "triangle":
        return (f"{x:.2f} {y + s:.2f} m {x - 0.866 * s:.2f} {y - 0.5 * s:.2f} l "
                f"{x + 0.866 * s:.2f} {y - 0.5 * s:.2f} l h f\n")
    if shape == "square":
        return f"{x - s:.2f} {y - s:.2f} {2 * s:.2f} {2 * s:.2f} re f\n"
    return (f"{x - s:.2f} {y - s:.2f} m {x + s:.2f} {y + s:.2f} l "
            f"{x - s:.2f} {y + s:.2f} m {x + s:.2f} {y - s:.2f} l S\n")


def write_pdf(path, content: bytes, alphas):
    stream = zlib.compress(content, 9)
    gstates = " ".join(f"/G{i} << /ca {a} /CA {a} >>" for i, a in enumerate(alphas))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH:.3f} {HEIGHT:.3f}] "
         f"/Resources << /ExtGState << {gstates} >> >> /Contents 4 0 R >>").encode(),
        f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


tables = {lvl: read(DATA / f"scatter_{lvl}.dat") for lvl, *_ in LEVELS}
for name, (xk, yk, xlog, ylog, (x0, x1), (y0, y1)) in PANELS.items():
    parts, drawn = [], {}
    for index, (lvl, shape, rgb, alpha, size) in enumerate(LEVELS):
        r, g, b = (c / 255 for c in rgb)
        parts.append(f"/G{index} gs {r:.4f} {g:.4f} {b:.4f} rg {r:.4f} {g:.4f} {b:.4f} RG 0.35 w\n")
        count = 0
        for row in tables[lvl]:
            if row[xk] == "nan" or row[yk] == "nan":
                continue
            x = scale(float(row[xk]), x0, x1, xlog, WIDTH)
            y = scale(float(row[yk]), y0, y1, ylog, HEIGHT)
            if x is None or y is None or not (-1 <= x <= WIDTH + 1 and -1 <= y <= HEIGHT + 1):
                continue
            parts.append(mark(shape, x, y, size))
            count += 1
        drawn[lvl] = count
    write_pdf(DATA / f"cloud_{name}.pdf", "".join(parts).encode(), [a for *_, a, _s in LEVELS])
    print(name, drawn)
