"""Draw the coordinate grid that gets painted on the wall.

One pixel per millimetre. Thin lines every 5 cm, thick lines with labels
every 10 cm, and the origin (u = 0, v = 0) at the bottom-left corner.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PX_PER_M = 1000


def make_wall_grid(width_m: float, height_m: float, out_path: Path) -> Path:
    w, h = round(width_m * PX_PER_M), round(height_m * PX_PER_M)
    img = Image.new("RGB", (w, h), (236, 232, 222))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow < 10.1 has no size argument
        font = ImageFont.load_default()

    def to_px(u: float, v: float) -> tuple[int, int]:
        # Image rows grow downwards, wall v grows upwards.
        return round(u * PX_PER_M), h - 1 - round(v * PX_PER_M)

    step = 0.05
    for i in range(int(width_m / step) + 1):
        u = i * step
        major = i % 2 == 0
        draw.line([to_px(u, 0), to_px(u, height_m)], fill=(90, 90, 90) if major else (175, 175, 175), width=4 if major else 2)
        if major and 0 < i:
            draw.text((to_px(u, 0)[0] + 6, h - 40), f"{round(u * 100)}", fill=(30, 30, 160), font=font)
    for j in range(int(height_m / step) + 1):
        v = j * step
        major = j % 2 == 0
        draw.line([to_px(0, v), to_px(width_m, v)], fill=(90, 90, 90) if major else (175, 175, 175), width=4 if major else 2)
        if major and 0 < j:
            draw.text((8, to_px(0, v)[1] + 4), f"{round(v * 100)}", fill=(160, 30, 30), font=font)

    # Axes from the origin.
    draw.line([to_px(0, 0), to_px(0.12, 0)], fill=(30, 30, 160), width=10)
    draw.line([to_px(0, 0), to_px(0, 0.12)], fill=(160, 30, 30), width=10)
    draw.text((to_px(0.13, 0.02)), "u (cm)", fill=(30, 30, 160), font=font)
    draw.text((to_px(0.02, 0.14)), "v (cm)", fill=(160, 30, 30), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
