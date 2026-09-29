"""Does an i4 landmark name agree with the HM3D category under its target? (CPU)

Used by scripts/audit_instruction_labels.py --mode semantic. HM3D categories are raw
annotator strings ("couch", "kitchen cabinet", "unknown"), so matching goes through a small
synonym table and word overlap rather than exact equality.
"""
import json

import numpy as np
from PIL import Image, ImageDraw

IGNORED = {"floor", "ceiling", "unknown", "carpet", "rug", "floor mat", "mat"}
# surfaces behind an object: its outline (floor to 1 m) often covers more of them than of the
# object, so they only count when the name itself says wall (spec §3)
BACKDROP = {"wall", "door frame", "window", "window frame"}
OPENING_WORDS = {"door", "doorway", "opening", "hallway", "corridor", "passage", "entrance",
                 "entry", "archway", "arch", "gap", "hall", "passageway"}
_SYNONYMS = [
    {"sofa", "couch", "sectional", "loveseat", "settee"},
    {"armchair", "chair", "seat"},
    {"table", "desk"},
    {"cabinet", "cupboard", "dresser", "sideboard", "drawers", "wardrobe", "closet"},
    {"counter", "countertop", "island"},
    {"shelf", "shelves", "bookshelf", "bookcase", "shelving"},
    {"column", "pillar", "post", "pole"},
    {"wall", "partition"},
    {"railing", "banister", "balustrade", "rail"},
    {"plant", "planter", "pot", "flowerpot"},
    {"bed", "mattress"},
    {"fridge", "refrigerator"},
    {"tv", "television", "monitor", "screen"},
    {"stairs", "staircase", "stair", "stairway", "steps"},
]
_CANON = {w: sorted(g)[0] for g in _SYNONYMS for w in g}


def canonical(word):
    w = word.lower().strip()
    if w in _CANON:
        return _CANON[w]
    for stem in (w[:-2] if w.endswith("es") else None, w[:-1] if w.endswith("s") else None):
        if stem and len(stem) > 2:                 # "curtains" -> "curtain", "benches" -> "bench"
            return _CANON.get(stem, stem)
    return w


def name_matches(name, category):
    nw = {canonical(w) for w in name.lower().split()}
    cw = {canonical(w) for w in category.lower().split()}
    return bool(nw & cw)


def _mask(shape, polygon):
    H, W = shape
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([(u * W, v * H) for u, v in polygon], fill=1)
    return np.asarray(im) > 0


def target_category(ids, categories, polygon, name=None):
    """Most frequent category inside the target outline, ignoring floor-like surfaces and,
    unless `name` is a wall, the walls/windows behind the object. None if nothing is left."""
    cats = json.loads(categories) if isinstance(categories, str) else categories
    inside = ids[_mask(ids.shape, polygon)]
    skip = IGNORED if name and canonical((name.split() or [""])[-1]) == canonical("wall") else IGNORED | BACKDROP
    counts = {}
    for i, n in zip(*np.unique(inside, return_counts=True)):
        c = cats.get(str(int(i)), "unknown")
        if c not in skip:
            counts[c] = counts.get(c, 0) + int(n)
    return max(counts, key=counts.get) if counts else None


def target_categories(ids, categories, polygon, name=None, min_frac=0.10):
    """Every category covering >= min_frac of the outline (same skips as target_category).
    Composite objects need this: a dining table's outline is mostly chair pixels."""
    cats = json.loads(categories) if isinstance(categories, str) else categories
    inside = ids[_mask(ids.shape, polygon)]
    skip = IGNORED if name and canonical((name.split() or [""])[-1]) == canonical("wall") else IGNORED | BACKDROP
    out = set()
    for i, n in zip(*np.unique(inside, return_counts=True)):
        c = cats.get(str(int(i)), "unknown")
        if c not in skip and n >= min_frac * max(len(inside), 1):
            out.add(c)
    return out


def beside_categories(ids, categories, bbox, grow=0.5, min_frac=0.02):
    """Categories covering >= min_frac of the bbox grown by `grow` of its size each side."""
    cats = json.loads(categories) if isinstance(categories, str) else categories
    H, W = ids.shape
    u0, v0, u1, v1 = bbox
    du, dv = (u1 - u0) * grow, (v1 - v0) * grow
    r0, r1 = int(max(0, v0 - dv) * H), int(min(1, v1 + dv) * H) + 1
    c0, c1 = int(max(0, u0 - du) * W), int(min(1, u1 + du) * W) + 1
    region = ids[r0:r1, c0:c1]
    out = set()
    for i, n in zip(*np.unique(region, return_counts=True)):
        c = cats.get(str(int(i)), "unknown")
        if c not in IGNORED and n >= min_frac * region.size:
            out.add(c)
    return out
