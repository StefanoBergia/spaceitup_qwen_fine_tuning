"""Route checker: an image-based check of d3 route texts for hard frames (spec
docs/superpowers/specs/2026-11-16-route-checker-design.md).

A vision-language model (Qwen3.6 via vLLM, `scripts/label_descriptions.py --stage check`) sees the frame as the
user saw it on the label audits (route in magenta, numbered outlines) plus the text, and answers only what the map
cannot see: do the names fit real things, does the route pass a big obstacle the text leaves out, is the stop phrase
natural. It never judges left/right (round i1: Cosmos-Nano was at chance on drawn-route sides; sides come exactly
from the metadata). The answers become a "probably wrong" score fitted on the user's audit marks; only d3 TRAIN
routes are ever dropped.
"""

import json
import re

from rover_vlm.traces import answer_payload, parse_trace

CHECK_PROMPT_VERSION = "k1"
_YN = {"yes": True, "true": True, "no": False, "false": False}
_FIT = ("yes", "no", "unsure")

_PROMPT_K1 = """You are checking a navigation instruction for a small ground rover against a picture of what the rover's \
camera sees.

The picture:
- The magenta line is the route the rover drives. It is solid where you can see it and dashed where it goes out of \
sight; a ring marks where it leaves the view.
- Numbered outlines mark objects the instruction may mention: the object the rover stops at is outlined in green, \
the objects it passes are outlined in orange.

The instruction:
"{text}"

Which side the rover passes things on (left or right) is already correct, taken from the map: do not judge left or \
right. Judge only these:
1. For every OUTLINED object (it has a number in the picture) that the instruction names: give its "number", the \
"name" the instruction uses for it, whether that name fits what is outlined ("fits"), and "is_object": answer "no" \
only when the instruction calls it an object or furniture but the outline is really part of a wall, floor or \
ceiling. Places are not objects: do not list walls, floors, doorways, corners, hallways or rooms that have no \
outline (e.g. "by the wall", "in the corner", "in the middle of the room"); a text that names no outlined object \
has an empty list.
2. Does the route pass close to a big obstacle (a pillar, a column, a large piece of furniture) that the instruction \
does not mention ("missing_obstacle")?
3. Is the way the instruction says where the rover stops a natural way to describe where the route ends \
("stop_ok")? A person might rather say e.g. "in the middle of the room" or "in front of the sofa"; if so, write \
that in "better_stop".
4. Overall, would a person who follows the instruction end where the route ends ("overall_true"), and how sure are \
you, 0 to 100 ("confidence")?

Reply with only this JSON:
{{"names": [{{"number": <n>, "name": "<as written in the instruction>", "fits": "yes|no|unsure", \
"is_object": "yes|no|unsure"}}], \
"missing_obstacle": "yes|no", "stop_ok": "yes|no", "better_stop": "<short phrase or empty>", \
"overall_true": "yes|no", "confidence": <0-100>}}"""


def check_prompt(text: str) -> str:
    """The k1 checker prompt for one instruction text."""
    return _PROMPT_K1.format(text=text)


def _yes_no(v):
    if isinstance(v, bool):
        return v
    return _YN.get(v.strip().lower()) if isinstance(v, str) else None


def _confidence(v):
    if v is None:
        return 50
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            return 50
        v = m.group(0)
    try:
        return int(round(min(100.0, max(0.0, float(v)))))
    except (TypeError, ValueError):
        return 50


def parse_check_reply(raw: str) -> dict | None:
    """The checker's JSON as a normalised dict, or None when it is absent or lacks a yes/no for missing_obstacle,
    stop_ok or overall_true. Tolerates a <think> block, a code fence and prose around the JSON."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is None:
        return None
    out = {}
    for k in ("missing_obstacle", "stop_ok", "overall_true"):
        out[k] = _yes_no(payload.get(k))
        if out[k] is None:
            return None
    names = []
    for n in payload.get("names") or []:
        if not isinstance(n, dict):
            continue
        fits, obj = (str(n.get(k, "unsure")).strip().lower() for k in ("fits", "is_object"))
        num = n.get("number")
        names.append({"number": int(num) if isinstance(num, (int, float)) or (isinstance(num, str) and num.isdigit())
                      else None, "name": str(n.get("name", "")), "fits": fits if fits in _FIT else "unsure",
                      "is_object": obj if obj in _FIT else "unsure"})
    better = payload.get("better_stop")
    return {"names": names, "missing_obstacle": out["missing_obstacle"], "stop_ok": out["stop_ok"],
            "better_stop": better.strip() if isinstance(better, str) else "",
            "overall_true": out["overall_true"], "confidence": _confidence(payload.get("confidence"))}


def check_route(ask, item: dict, seed: int) -> dict:
    """Check one manifest row ({id, image, text}) with up to two calls (seed, seed + 1). Two unreadable replies give a
    row with rejects ["parse"] (a result, not an error, so --resume does not retry it forever). Never raises on a bad
    reply; transport errors propagate to the caller."""
    prompt = check_prompt(item["text"])
    row = {"id": item["id"], "prompt_version": CHECK_PROMPT_VERSION, "rejects": [], "attempts": 0, "finish": None}
    raws, fin = [], None
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, fin = ask(item["image"], prompt, s)
        raws.append(raw)
        parsed = parse_check_reply(raw)
        if parsed is not None:
            return {**row, "attempts": attempt, "finish": fin, "passed": True, **parsed}
    return {**row, "attempts": 2, "finish": fin, "passed": False, "rejects": ["parse"], "raw": raws}


# --- calibration items from the saved audit pages ---------------------------------------------------------------
_PAGE_DATA = re.compile(r"const DATA = (\{.*?\});\n", re.S)


def audit_page_items(html: str) -> list[dict]:
    """The items of a saved audit page (the payload scripts/audit_instruction_labels.py writes as `const DATA = {...};`)."""
    m = _PAGE_DATA.search(html)
    if not m:
        raise ValueError("no `const DATA = {...};` payload in this page")
    return json.loads(m.group(1))["items"]


def mark_wrong(verdict) -> bool | None:
    """A user mark as wrong (True) / right (False) / not counted (None: unsure or missing)."""
    if verdict in (None, "unsure", "?"):
        return None
    return verdict != "ok"


def calibration_rows(slug: str, items: list[dict], marks: dict, image_dir) -> list[dict]:
    """Manifest rows for one audit page: every item with a counted mark, except drop items ("DROPPED ..." texts have
    no route text to check). The page's own JPEG (exactly what the user judged) goes to image_dir/<slug>/<id>.jpg.
    Ids are "<slug>/<route_id>", so a route judged on two pages is two rows."""
    from base64 import b64decode
    from pathlib import Path
    out_dir = Path(image_dir) / slug
    rows = []
    for it in items:
        m = marks.get(it["id"])
        wrong = mark_wrong(m.get("verdict")) if m else None
        if wrong is None or it["instruction"].startswith("DROPPED"):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{it['id']}.jpg"
        path.write_bytes(b64decode(it["img"].split(",", 1)[1]))
        rows.append({"id": f"{slug}/{it['id']}", "route_id": it["id"], "page": slug, "image": str(path.resolve()),
                     "text": it["instruction"], "wrong": wrong, "verdict": m["verdict"], "note": m.get("note", "")})
    return rows


# --- the score: features, a small logistic fit, out-of-fold scores, AUC, the removal curve ----------------------
import math

import numpy as np

FEATURES = ("name_no", "object_no", "missing_obstacle", "stop_no", "overall_no", "p_false")


def features(row: dict):
    """The checker's answers as a float vector (FEATURES order), or None for a row without a readable reply."""
    if not row.get("passed"):
        return None
    names = row.get("names") or []
    c = row.get("confidence", 50) / 100.0
    return np.array([float(any(n["fits"] == "no" for n in names)), float(any(n["is_object"] == "no" for n in names)),
                     float(row["missing_obstacle"]), float(not row["stop_ok"]), float(not row["overall_true"]),
                     round(1.0 - c if row["overall_true"] else c, 6)])


def fit_logistic(X, y, l2: float = 1.0, iters: int = 50):
    """L2-regularised logistic regression by Newton steps (no sklearn); returns weights with the bias last."""
    X1 = np.column_stack([np.asarray(X, float), np.ones(len(X))])
    y = np.asarray(y, float)
    w = np.zeros(X1.shape[1])
    reg = l2 * np.eye(X1.shape[1])
    reg[-1, -1] = 0.0                          # the bias is not shrunk
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X1 @ w))
        g = X1.T @ (p - y) + reg @ w
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    """P(wrong) under weights from fit_logistic."""
    X1 = np.column_stack([np.asarray(X, float), np.ones(len(X))])
    return 1.0 / (1.0 + np.exp(-X1 @ np.asarray(w)))


def group_folds(groups, k: int, seed: int) -> list[int]:
    """A fold per row such that rows of one group (a place, scene_cXXX) share a fold."""
    uniq = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[i]: n % k for n, i in enumerate(order)}
    return [fold_of[g] for g in groups]


def oof_scores(X, y, groups, k: int = 5, seed: int = 0):
    """Out-of-fold P(wrong): each row scored by a fit that never saw its place."""
    X, y = np.asarray(X, float), np.asarray(y, bool)
    folds = np.array(group_folds(groups, k, seed))
    out = np.zeros(len(y))
    for f in range(k):
        te = folds == f
        if te.any():
            out[te] = predict(fit_logistic(X[~te], y[~te]), X[te])
    return out


def auc(scores, wrong) -> float:
    """Area under the ROC curve of `scores` for the `wrong` rows (ties averaged); nan without both classes."""
    s, y = np.asarray(scores, float), np.asarray(wrong, bool)
    if y.all() or not y.any():
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = ((np.cumsum(cnt) - cnt) + (cnt + 1) / 2.0)[inv]
    npos = y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (~y).sum()))


def removal_curve(scores, wrong, shares=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)) -> list[dict]:
    """For each target share: the threshold = the score of the ceil(share * n)-th highest row, and what
    `score >= threshold` (exactly what --apply does) removes: the count and actual share (ties can push it past the
    target), the accuracy of what is kept, and the share of the wrong rows caught."""
    s, y = np.asarray(scores, float), np.asarray(wrong, bool)
    desc = np.sort(s)[::-1]
    out = []
    for share in shares:
        k = math.ceil(share * len(s))
        thr = float(desc[k - 1]) if k else float("inf")
        gone = s >= thr
        out.append({"share": share, "threshold": thr, "removed": int(gone.sum()),
                    "share_actual": round(float(gone.mean()), 4),
                    "kept_accuracy": round(float(1 - y[~gone].mean()) if (~gone).any() else 1.0, 4),
                    "wrong_caught": round(float(y[gone].sum() / y.sum()) if y.any() else 0.0, 4)})
    return out


def drop_records(recs: list[dict], ids: set) -> tuple[list[dict], int]:
    """(records whose id is not in `ids`, how many were dropped). finalize-descriptions applies it to TRAIN only."""
    kept = [r for r in recs if r["id"] not in ids]
    return kept, len(recs) - len(kept)


# the words a text uses for each d3 stop kind (route_events.stop_kind names), checked after its last "stop"
_KIND_WORDS = {"object:next to": ("next to", "beside"), "object:in front of": ("in front of",),
               "wall:in front of": ("in front of",), "object:behind": ("behind",), "place:wall": ("wall",),
               "place:corner": ("corner",), "place:doorway": ("doorway", "door"), "place:hallway_end": ("end of",),
               "place:hallway": ("hallway", "corridor"), "place:room_middle": ("middle",)}


def kind_matches_text(kind: str, text: str) -> bool:
    """Whether `text`'s stop clause (after its last "stop", else the whole text) uses the words of `kind`. The
    calibration report bins a d3 row under its current stop kind only when it does: pages judged before an events
    rebuild (d3_g3, before Addendum A2) can hold texts for a stop the rebuild changed."""
    t = text.lower()
    i = t.rfind("stop")
    clause = t[i:] if i >= 0 else t
    return any(w in clause for w in _KIND_WORDS.get(kind, ()))
