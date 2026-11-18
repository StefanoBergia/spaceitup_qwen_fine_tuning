"""Rewrite a real-image eval set into a newer prompt framing.

The real sets under data/prepared_real/<set>/ were built with the legacy prompt
("the goal is located straight ahead", answer carries a "goal" key). Adapters from round 3
("goal") and round 4 ("endpoints") were trained on prompts that *name* the goal, or the
route's entry point and the goal, and answer with the path alone. Scoring them on the legacy
records would test a prompt they never saw, so this script re-renders the records the way
scripts/prepare_habitat.py does for Habitat: the entry point is the label's first waypoint,
the goal its last, each with its visibility flag; the answer drops the "goal" key.

CPU only, seconds. Writes data/prepared_real/<set>_<framing>/{eval.json,meta.json}; the source
set is untouched, so every result attached to it stays valid.

    uv run scripts/reframe_real_eval.py --set real_clips --framing endpoints
    uv run scripts/reframe_real_eval.py --set tum_pioneer gnd_campus real_clips --framing endpoints
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rover_vlm.habitat_data import FRAMINGS, _r3, endpoints_prompt, goal_prompt  # noqa: E402
from rover_vlm.eval import parse_path_answer  # noqa: E402


def reframe_record(rec, framing):
    gt = parse_path_answer(rec["conversations"][1]["value"])
    if gt is None or not gt["path"]:
        return None
    path = [[_r3(x), _r3(y), int(v)] for x, y, v in gt["path"]]
    goal = [_r3(gt["goal"][0]), _r3(gt["goal"][1]), int(gt["goal"][2])]
    # The legacy label's goal is a route point; make the path end exactly on it so the
    # "last waypoint is the goal" contract holds (prepare_habitat asserts the same).
    if path[-1][:2] != goal[:2]:
        path.append(goal)
    start = path[0]
    if framing == "goal":
        human = goal_prompt(goal)
    elif framing == "endpoints":
        human = endpoints_prompt(start, goal)
    else:
        raise ValueError(framing)
    out = dict(rec)
    out["conversations"] = [
        {"from": "human", "value": human},
        {"from": "gpt", "value": json.dumps({"path": path}, separators=(",", ":"))},
    ]
    meta = dict(rec.get("habitat_meta") or {})
    meta.update({
        "framing": framing, "goal_in_prompt": True,
        "goal_uv_norm": goal[:2], "goal_visible": goal[2],
        "start_uv_norm": start[:2], "entry_edge": _entry_edge(start),
    })
    out["habitat_meta"] = meta
    return out


def _entry_edge(start, tol=0.02):
    x, y = start[0], start[1]
    if y >= 1.0 - tol:
        return "bottom"
    if x <= tol:
        return "left"
    if x >= 1.0 - tol:
        return "right"
    return "interior"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", nargs="+", required=True, help="names under data/prepared_real/")
    p.add_argument("--framing", choices=[f for f in FRAMINGS if f != "legacy"], default="endpoints")
    p.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "prepared_real")
    args = p.parse_args()
    for name in args.set:
        src = args.root / name
        dst = args.root / f"{name}_{args.framing}"
        recs = json.load(open(src / "eval.json"))
        out, dropped = [], 0
        for rec in recs:
            r = reframe_record(rec, args.framing)
            if r is None:
                dropped += 1
                continue
            out.append(r)
        dst.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(dst / "eval.json", "w"))
        if (src / "meta.json").exists():
            meta = json.load(open(src / "meta.json"))
            meta["reframed_from"] = name
            meta["framing"] = args.framing
            json.dump(meta, open(dst / "meta.json", "w"), indent=2)
        for extra in ("images",):
            if (src / extra).exists() and not (dst / extra).exists():
                (dst / extra).symlink_to((src / extra).resolve())
        edges = {}
        for r in out:
            e = r["habitat_meta"]["entry_edge"]
            edges[e] = edges.get(e, 0) + 1
        print(f"{name} -> {dst.relative_to(REPO_ROOT)}: {len(out)} records"
              f" ({dropped} unparsable dropped), entry edges {edges}")
        print("  example prompt:", out[0]["conversations"][0]["value"][8:160].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
