"""Drawing helpers shared by the HTML report generators.

Predicted paths are drawn over cluttered indoor photographs, so every stroke gets a
white underlay — a thin coloured line on its own disappears against furniture. Waypoint
visibility is encoded by fill rather than by a second colour: filled = the model called
that point visible, hollow = obstructed. That keeps the colour channel free to mean
"which model" and leaves visibility readable in greyscale.
"""

import base64
import io

from PIL import Image, ImageDraw

WHITE = (255, 255, 255)
VIS_GREEN = (0, 160, 0)
OBS_RED = (208, 59, 59)
GT_GREY = (105, 105, 105)


def embed_jpeg(img, max_px=480, quality=80):
    """Downscale and inline an image as a data URI (artifact CSP blocks external hosts)."""
    img = img.convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def draw_polyline(draw, pts, color, W, H, width=4, underlay=True):
    """Draw normalized [(x, y), ...] as pixels, with a white halo for contrast."""
    xy = [(x * W, y * H) for x, y in pts]
    if len(xy) < 2:
        return xy
    if underlay:
        draw.line(xy, fill=WHITE, width=width + 4, joint="curve")
    draw.line(xy, fill=color, width=width, joint="curve")
    return xy


def draw_path(draw, pts, color, W, H, r=6, width=4):
    """Polyline of [x, y, v] waypoints; filled marker = visible, hollow = obstructed."""
    xy = draw_polyline(draw, [(p[0], p[1]) for p in pts], color, W, H, width)
    if not xy:
        xy = [(p[0] * W, p[1] * H) for p in pts]
    for (x, y), p in zip(xy, pts):
        box = [x - r, y - r, x + r, y + r]
        if p[2] == 1:
            draw.ellipse(box, fill=color, outline=WHITE, width=2)
        else:
            draw.ellipse(box, fill=WHITE, outline=color, width=3)


def draw_goal(draw, goal, W, H, r=9):
    """Goal marker coloured by *predicted* visibility — green visible, red obstructed."""
    x, y = goal[0] * W, goal[1] * H
    draw.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=WHITE)
    draw.ellipse([x - r, y - r, x + r, y + r],
                 fill=VIS_GREEN if goal[2] == 1 else OBS_RED, outline=WHITE, width=2)


def side_by_side(left, right, gap=8):
    """Join two equal-height panels with a light gutter."""
    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + right.width + gap, h), WHITE)
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    return out


def faint(color, mix=0.55):
    """The model's colour blended towards white — for sampled draws behind the greedy path."""
    return tuple(int(c + (255 - c) * mix) for c in color)


def render_pair(image_path, gt, pred_left, pred_right, color_left, color_right, max_px=520,
                samples_left=None, samples_right=None):
    """One frame rendered twice — ground truth plus each model's prediction — joined.

    Returns a PIL image. `pred_*` may be None (the model produced nothing parseable),
    in which case that panel shows ground truth alone rather than being dropped, so the
    failure stays visible instead of silently vanishing from the gallery.

    `samples_*` (optional) are parsed predictions from sampled draws of the same model;
    they are drawn as thin, faint lines UNDER the greedy path so the panel shows how much
    the answer moves between draws. None entries (unparseable draws) are skipped.
    """
    panels = []
    for pred, color, samples in ((pred_left, color_left, samples_left),
                                 (pred_right, color_right, samples_right)):
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        if gt and gt.get("path"):
            draw_polyline(d, [(p[0], p[1]) for p in gt["path"]], GT_GREY, W, H, width=9)
        for s in samples or ():
            pts = (s.get("path") or [s["goal"]]) if s else None
            if pts:
                draw_polyline(d, [(p[0], p[1]) for p in pts], faint(color), W, H,
                              width=2, underlay=False)
        if pred and pred.get("path"):
            draw_path(d, pred["path"], color, W, H)
        if pred and pred.get("goal"):
            draw_goal(d, pred["goal"], W, H)
        if max(img.size) > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        panels.append(img)
    return side_by_side(*panels)
