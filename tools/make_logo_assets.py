"""Cut the logo sheet into website-ready PNGs with a transparent background.

    python tools/make_logo_assets.py frontend/assets/logo-sheet.png frontend/assets
"""
import sys
from pathlib import Path

from PIL import Image

# (name, left, top, right, bottom) in pixels on the 1254 x 1254 sheet
CROPS = [
    ("logo-mark", 395, 80, 850, 460),
    ("logo-full", 255, 80, 1000, 600),
    ("app-icon", 62, 708, 268, 914),
]


def to_transparent(img, low=10, high=46):
    """Near-black background -> transparent, with a soft edge so the glow survives."""
    rgba = img.convert("RGBA")
    px = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, _ = px[x, y]
            m = max(r, g, b)
            a = 0 if m <= low else 255 if m >= high else int(255 * (m - low) / (high - low))
            px[x, y] = (r, g, b, a)
    return rgba


def main(sheet, out_dir):
    img = Image.open(sheet).convert("RGB")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, l, t, r, b in CROPS:
        crop = img.crop((l, t, r, b))
        target = out / f"{name}.png"
        (to_transparent(crop) if name != "app-icon" else crop).save(target)
        print("saved", target, crop.size)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])