"""Habitat path *classification*: render candidate paths onto the frame and ask
the model which one is traversable.

Companion to habitat_data.py (which does path *regression*). Same source layout:
<dataset_root>/<scene>/samples/<scene>_cNNN/ with fpv_enhanced.png, fpv_paths.json,
meta.json.

Each sample offers several candidate paths (3 or 5, occasionally 2 or 4). meta.json
gives `label` (the canonical correct index) and `accepted` (every acceptable index).
Verified over all 4,398 samples: `label` is always in `accepted`, `accepted` is
exactly the set of `feasible` candidates, and `candidates[i]["display_index"] == i`
— so the badge number drawn on the image *is* the dataset index, no conversion.

Coordinates in fpv_paths.json live in `image_size` space (512x512), while
fpv_enhanced.png is 1024x1024 — normalize by image_size, never by pixel size.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .habitat_data import DATASET_ROOT, IMAGE_NAME, clip_polyline_unit, normalize_points

# Okabe-Ito colorblind-safe palette; index i -> candidate i. Color is never the only
# channel — every candidate also carries a numbered badge.
CANDIDATE_COLORS = [
    (230, 159, 0),    # orange
    (86, 180, 233),   # sky blue
    (0, 158, 115),    # bluish green
    (213, 94, 0),     # vermillion
    (204, 121, 167),  # reddish purple
]
GOAL_COLOR = (255, 255, 255)
RENDER_SIZE = 768

CHOICE_PROMPT = (
    "<image>\n"
    "You are a rover navigating an indoor environment. Several candidate paths to the "
    "goal have been drawn on the image, each marked with a numbered badge. Exactly one "
    "of them should be chosen: the path that is actually traversable and does not run "
    "through walls, furniture or other obstacles. Reply with the number of that path as "
    'JSON: {"choice": N}.'
)


def candidate_polylines(fpv_paths):
    """Every candidate's runs flattened, normalized by image_size, clipped to [0,1]^2.

    Returns a list (one entry per candidate) of [(x, y, hidden), ...]. A candidate whose
    path lies entirely outside the frame yields an empty list.
    """
    w, h = fpv_paths["image_size"]
    out = []
    for cand in fpv_paths.get("candidates", []):
        raw = [
            (float(u), float(v), bool(run["hidden"]))
            for run in cand.get("runs", [])
            for (u, v) in run["uv"]
        ]
        out.append(clip_polyline_unit(normalize_points(raw, w, h)) if len(raw) >= 2 else [])
    return out


def _point_at_fraction(pts, frac):
    """Point at `frac` of the way along a polyline's arc length."""
    if len(pts) == 1:
        return pts[0][0], pts[0][1]
    seg = [
        ((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
        for i in range(len(pts) - 1)
    ]
    total = sum(seg)
    if total <= 0:
        return pts[0][0], pts[0][1]
    want, acc = frac * total, 0.0
    for i, s in enumerate(seg):
        if acc + s >= want and s > 0:
            t = (want - acc) / s
            return (pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                    pts[i][1] + t * (pts[i + 1][1] - pts[i][1]))
        acc += s
    return pts[-1][0], pts[-1][1]


# badge geometry in normalized units: radius 0.022 -> keep centres ~1.6 diameters apart
_BADGE_MARGIN = 0.045
_BADGE_MIN_SEP = 0.075
_BADGE_FRACTIONS = (0.5, 0.38, 0.62, 0.28, 0.72, 0.2, 0.8, 0.14, 0.86)


def badge_positions(fpv_paths, polys):
    """Where to draw each candidate's numbered badge, in normalized coords.

    The dataset's own `badge_uv` falls outside the frame for ~32% of candidates and
    collides with a neighbour often enough to make badges ambiguous, so it is used only
    when it is both in-frame and clear; otherwise the badge is placed on the candidate's
    own clipped polyline, as far from already-placed badges as possible. A candidate
    with nothing left inside the frame gets None.
    """
    w, h = fpv_paths["image_size"]
    cands = fpv_paths.get("candidates", [])
    placed: list[tuple[float, float]] = []
    out: list[tuple[float, float] | None] = []

    def clear(p):
        return all(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 >= _BADGE_MIN_SEP
                   for q in placed)

    def in_frame(p):
        return _BADGE_MARGIN <= p[0] <= 1 - _BADGE_MARGIN and _BADGE_MARGIN <= p[1] <= 1 - _BADGE_MARGIN

    for i, cand in enumerate(cands):
        pts = polys[i] if i < len(polys) else []
        if not pts:
            out.append(None)
            continue
        options = []
        bu, bv = cand.get("badge_uv", (None, None))
        if bu is not None:
            options.append((bu / w, bv / h))
        options += [_point_at_fraction(pts, f) for f in _BADGE_FRACTIONS]

        viable = [p for p in options if in_frame(p)]
        if not viable:  # polyline is in-frame but hugs an edge; clamp it inward
            viable = [(min(max(p[0], _BADGE_MARGIN), 1 - _BADGE_MARGIN),
                       min(max(p[1], _BADGE_MARGIN), 1 - _BADGE_MARGIN)) for p in options]
        pick = next((p for p in viable if clear(p)), None)
        if pick is None:  # everything collides — take the least-bad spot
            pick = max(viable, key=lambda p: min(
                (((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 for q in placed), default=1.0))
        placed.append(pick)
        out.append(pick)
    return out


def drawable_candidates(fpv_paths):
    """Indices whose path actually appears inside the frame (>= 2 clipped points)."""
    return [i for i, pts in enumerate(candidate_polylines(fpv_paths)) if len(pts) >= 2]


DASH_ON, DASH_OFF = 13, 9


def flag_runs(pts):
    """Split a [(x, y, hidden), ...] polyline into maximal same-visibility stretches.

    Returns [(hidden, [(x, y), ...]), ...]. A segment takes the flag of its *first*
    point, matching the convention in habitat_data.clip_polyline_unit.
    """
    runs = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if runs and runs[-1][0] == a[2]:
            runs[-1][1].append((b[0], b[1]))
        else:
            runs.append((a[2], [(a[0], a[1]), (b[0], b[1])]))
    return runs


def dashify(pts, on=DASH_ON, off=DASH_OFF):
    """Approximate a dashed line: split a polyline into the 'on' sub-polylines.

    Dash phase carries across vertices so corners don't reset the pattern.
    """
    if len(pts) < 2:
        return []
    out, cur, drawing, remaining = [], [pts[0]], True, on
    px, py = pts[0]
    for qx, qy in pts[1:]:
        seg = math.hypot(qx - px, qy - py)
        travelled = 0.0
        while seg - travelled > remaining:
            travelled += remaining
            t = travelled / seg
            mx, my = px + (qx - px) * t, py + (qy - py) * t
            if drawing:
                cur.append((mx, my))
                if len(cur) >= 2:
                    out.append(cur)
                cur = []
            else:
                cur = [(mx, my)]
            drawing = not drawing
            remaining = on if drawing else off
        remaining -= seg - travelled
        if drawing:
            cur.append((qx, qy))
        px, py = qx, qy
    if drawing and len(cur) >= 2:
        out.append(cur)
    return out


def _font(size):
    """A legible bitmap/truetype font, whatever this box has."""
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:  # Pillow >= 10.1 can scale the built-in font
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_badge(draw, xy, text, color, font, r):
    """Filled circle with a white ring and dark centred number."""
    x, y = xy
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255), width=3)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (x - (box[2] - box[0]) / 2 - box[0], y - (box[3] - box[1]) / 2 - box[1]),
        text, fill=(0, 0, 0), font=font,
    )


def render_choice_image(sample_dir, out_path, size=RENDER_SIZE, highlight=None, dashed=False):
    """Draw every candidate path + numbered badge on fpv_enhanced.png -> out_path (JPEG).

    `highlight` (an index) thickens that candidate and `dashed` breaks occluded stretches
    into dashes. Both are **inspection aids only** — training composites are rendered with
    the defaults (solid, no highlight), because dashing encodes occlusion that the model
    would otherwise have to infer, which would change the task. Keep the defaults if you
    are generating data. Returns the number of candidates drawn.
    """
    sample_dir, out_path = Path(sample_dir), Path(out_path)
    fpv = json.loads((sample_dir / "fpv_paths.json").read_text())

    img = Image.open(sample_dir / IMAGE_NAME).convert("RGB")
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = _font(max(14, int(W * 0.028)))

    polys = candidate_polylines(fpv)
    for i, pts in enumerate(polys):
        if len(pts) < 2:
            continue
        color = CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
        base = 7 if highlight == i else 5
        # solid where the path is on visible ground, dashed where it passes behind an
        # obstacle — the underlay is dashed too, or a solid white line would show
        # through the gaps and read as a continuous path
        if dashed:
            strokes = []
            for hidden, run in flag_runs(pts):
                xy = [(x * W, y * H) for x, y in run]
                strokes += dashify(xy) if hidden else [xy]
        else:
            # one call for the whole polyline: splitting it changes how PIL renders the
            # joints at interior vertices, and training composites must stay reproducible
            strokes = [[(x * W, y * H) for x, y, _ in pts]]
        # every underlay first: interleaving them would let one run's white halo
        # paint over the previous run's colored line at the visibility boundary
        for s in strokes:
            draw.line(s, fill=(255, 255, 255), width=base + 4, joint="curve")
        for s in strokes:
            draw.line(s, fill=color, width=base, joint="curve")

    gw, gh = fpv["image_size"]
    gx, gy = fpv["goal"]["uv"][0] / gw * W, fpv["goal"]["uv"][1] / gh * H
    gr = max(6, int(W * 0.012))
    draw.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=GOAL_COLOR, outline=(0, 0, 0), width=2)

    # badges last, so they sit on top of every polyline
    badge_r = max(12, int(W * 0.022))
    for i, pos in enumerate(badge_positions(fpv, polys)):
        if pos is None:
            continue
        _draw_badge(
            draw,
            (pos[0] * W, pos[1] * H),
            str(i),
            CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)],
            font,
            badge_r,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)
    return len(polys)


def format_choice_answer(index):
    return json.dumps({"choice": int(index)}, separators=(",", ":"))


def build_choice_record(sample_dir, image_path, meta=None):
    """Habitat sample dir + an already-rendered image -> conversation record, or None.

    `choice_meta` carries the scoring sidecar (accepted set, candidate kinds) that the
    answer string itself cannot express. TrajectoryDataset reads only id/image/
    conversations, so this extra key is invisible to training.
    """
    sample_dir = Path(sample_dir)
    meta = meta if meta is not None else json.loads((sample_dir / "meta.json").read_text())
    cands = meta.get("candidates") or []
    label = meta.get("label")
    if label is None or not (0 <= label < len(cands)):
        return None
    # the whole scheme rests on badge number == dataset index; refuse to guess if not
    if any(c.get("display_index") != i for i, c in enumerate(cands)):
        return None
    accepted = sorted(int(a) for a in meta.get("accepted", []) if 0 <= int(a) < len(cands))
    if label not in accepted:
        return None

    return {
        "id": sample_dir.name,
        "image": [str(Path(image_path).resolve())],
        "conversations": [
            {"from": "human", "value": CHOICE_PROMPT},
            {"from": "gpt", "value": format_choice_answer(label)},
        ],
        "choice_meta": {
            "label": int(label),
            "accepted": accepted,
            "n_candidates": len(cands),
            "kinds": [c.get("label") for c in cands],
            "margin": meta.get("margin"),
            "near_symmetric": bool(meta.get("near_symmetric")),
        },
    }


__all__ = [
    "CANDIDATE_COLORS",
    "CHOICE_PROMPT",
    "DATASET_ROOT",
    "RENDER_SIZE",
    "badge_positions",
    "build_choice_record",
    "candidate_polylines",
    "drawable_candidates",
    "format_choice_answer",
    "render_choice_image",
]
