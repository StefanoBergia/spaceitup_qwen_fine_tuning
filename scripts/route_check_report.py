"""Route checker report (spec docs/superpowers/specs/2026-11-16-route-checker-design.md): how well the checker's answers
match the user's audit marks, and which routes a cutoff would drop.

Calibration (CPU, after slurm/check_routes.sbatch on the calibration manifest):
    uv run scripts/route_check_report.py --manifest outputs/route_check/calibration/manifest.jsonl \
        --check outputs/route_check/calibration/check_k1.jsonl --out outputs/route_check/calibration
  -> report.json + a printed table. The primary set is the four text-audit pages (the production format): a logistic
  score of the answers is fitted with place-grouped 5-fold cross-validation and its out-of-fold AUC is checked against
  the bar (>= 0.75 overall, > 0.6 within each d3 stop kind with >= 20 items); the plain "overall_true" confidence is
  the no-fit baseline. The two G1 stop pages are a secondary check of the stop answer only. Weights fitted on all
  primary rows are saved for --apply.

Apply (after the production check job):
    uv run scripts/route_check_report.py --apply outputs/route_check/train/check_k1.jsonl \
        --weights outputs/route_check/calibration/report.json --threshold T --out-ids outputs/route_check/train/drop_ids.txt
  -> the ids scoring >= T, one per line. Rows without a readable reply are never listed (the checker could not judge).
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from rover_vlm.route_check import FEATURES, auc, features, fit_logistic, oof_scores, predict, removal_curve
from rover_vlm.route_events import stop_kind

V5 = Path("data/prepared_habitat_instr_v5")
TEXT_PAGES = ("d2_g3", "d2_g3n2", "d3_g3", "d3_g3a2")
BAR_AUC, BAR_KIND_AUC, BAR_KIND_MIN = 0.75, 0.6, 20


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def place(route_id):
    return re.match(r"(\d{5}-[A-Za-z0-9]{11}_c\d{3})", route_id).group(1)


def check_rows(path):
    """Latest non-error row per id."""
    out = {}
    for r in jsonl(path):
        if not r.get("error"):
            out[r["id"]] = r
    return out


def stop_kinds(events_dir):
    kinds = {}
    for s in ("train", "eval"):
        for r in jsonl(events_dir / f"events_{s}.jsonl"):
            if r.get("reason") is None:
                kinds[r["id"]] = stop_kind(r["stop"]) or r["stop"]["type"]
    return kinds


def calibrate(args):
    man = {r["id"]: r for r in jsonl(args.manifest)}
    chk = check_rows(args.check)
    missing = [i for i in man if i not in chk]
    rows = [(man[i], chk[i]) for i in man if i in chk]
    unread = [m["id"] for m, c in rows if features(c) is None]
    rows = [(m, c) for m, c in rows if features(c) is not None]
    print(f"manifest {len(man)}; checked {len(man) - len(missing)} (missing {len(missing)}); unreadable {len(unread)}")
    prim = [(m, c) for m, c in rows if m["page"] in TEXT_PAGES]
    sec = [(m, c) for m, c in rows if m["page"] not in TEXT_PAGES]
    X = np.array([features(c) for _, c in prim])
    y = np.array([m["wrong"] for m, _ in prim])
    groups = [place(m["route_id"]) for m, _ in prim]
    fitted = oof_scores(X, y, groups, k=5, seed=0)
    base = X[:, FEATURES.index("p_false")]
    rep = {"n_primary": len(prim), "wrong_primary": int(y.sum()), "unreadable": len(unread), "missing": len(missing),
           "auc": {"fitted_oof": auc(fitted, y), "baseline_p_false": auc(base, y)},
           "per_feature_auc": {f: auc(X[:, j], y) for j, f in enumerate(FEATURES)}}
    rep["per_page"] = {p: {"n": int(sum(m["page"] == p for m, _ in prim)),
                           "auc": auc(fitted[[m["page"] == p for m, _ in prim]], y[[m["page"] == p for m, _ in prim]])}
                       for p in TEXT_PAGES}
    kinds = stop_kinds(args.events_dir)
    kind_of = [kinds.get(m["route_id"]) if m["page"].startswith("d3") else None for m, _ in prim]
    rep["per_stop_kind_d3"] = {}
    for k, n in Counter(k for k in kind_of if k).most_common():
        sel = np.array([kk == k for kk in kind_of])
        rep["per_stop_kind_d3"][k] = {"n": n, "wrong": int(y[sel].sum()), "auc": auc(fitted[sel], y[sel])}
    ok = y == 0
    rep["per_verdict"] = {}
    for v, n in Counter(m["verdict"] for m, _ in prim if m["wrong"]).most_common():
        sel = np.array([m["verdict"] == v for m, _ in prim]) | ok
        rep["per_verdict"][v] = {"n": n, "auc": auc(fitted[sel], y[sel])}
    rep["curve"] = removal_curve(fitted, y)
    big = {k: v for k, v in rep["per_stop_kind_d3"].items() if v["n"] >= BAR_KIND_MIN and 0 < v["wrong"] < v["n"]}
    rep["bar"] = {"auc": BAR_AUC, "kind_auc": BAR_KIND_AUC, "kinds_checked": sorted(big),
                  "pass": bool(rep["auc"]["fitted_oof"] >= BAR_AUC and all(v["auc"] > BAR_KIND_AUC for v in big.values()))}
    if sec:
        Xs = np.array([features(c) for _, c in sec])
        ys = np.array([m["wrong"] for m, _ in sec])
        rep["g1_secondary"] = {"n": len(sec), "wrong": int(ys.sum()),
                               "auc_stop_no": auc(Xs[:, FEATURES.index("stop_no")], ys),
                               "auc_p_false": auc(Xs[:, FEATURES.index("p_false")], ys)}
    rep["features"] = list(FEATURES)
    rep["weights"] = fit_logistic(X, y).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(rep, indent=2))
    scored = sorted(zip(fitted, prim), key=lambda t: -t[0])
    with (args.out / "scored.jsonl").open("w") as f:
        for s, (m, c) in scored:
            f.write(json.dumps({"id": m["id"], "score": round(float(s), 4), "wrong": m["wrong"], "verdict": m["verdict"],
                                "note": m.get("note", ""), "text": m["text"], "image": m["image"],
                                "better_stop": c.get("better_stop", ""), "names": c.get("names", []),
                                "missing_obstacle": c["missing_obstacle"], "stop_ok": c["stop_ok"],
                                "overall_true": c["overall_true"], "confidence": c["confidence"]}) + "\n")
    print(f"\nprimary (text pages): n={rep['n_primary']} wrong={rep['wrong_primary']}")
    print(f"AUC fitted (out-of-fold) {rep['auc']['fitted_oof']:.3f}   baseline p_false {rep['auc']['baseline_p_false']:.3f}")
    print("per feature: " + "  ".join(f"{k} {v:.3f}" for k, v in rep["per_feature_auc"].items()))
    print("per page:    " + "  ".join(f"{k} {v['auc']:.3f} (n={v['n']})" for k, v in rep["per_page"].items()))
    print("per d3 stop kind: " + "  ".join(f"{k} {v['auc']:.3f} ({v['wrong']}/{v['n']})"
                                          for k, v in rep["per_stop_kind_d3"].items()))
    print("per verdict: " + "  ".join(f"{k} {v['auc']:.3f} (n={v['n']})" for k, v in rep["per_verdict"].items()))
    if sec:
        g = rep["g1_secondary"]
        print(f"G1 (secondary): n={g['n']} wrong={g['wrong']} AUC stop_no {g['auc_stop_no']:.3f} p_false {g['auc_p_false']:.3f}")
    print("\nremove share -> kept accuracy, wrong caught:")
    for c in rep["curve"]:
        print(f"  {c['share']:.0%} ({c['removed']}) -> {c['kept_accuracy']:.1%} kept right, {c['wrong_caught']:.0%} of wrong caught"
              f"  (threshold {c['threshold']:.3f})")
    print(f"\nBAR (AUC >= {BAR_AUC} and > {BAR_KIND_AUC} within {rep['bar']['kinds_checked']}): "
          f"{'PASS' if rep['bar']['pass'] else 'FAIL'}")
    print(f"-> {args.out / 'report.json'}, {args.out / 'scored.jsonl'}")


def apply(args):
    rep = json.loads(args.weights.read_text())
    if rep.get("features") != list(FEATURES):
        raise SystemExit(f"{args.weights} was fitted on features {rep.get('features')}, not {list(FEATURES)}")
    chk = check_rows(args.apply)
    ids, X = zip(*[(i, features(c)) for i, c in chk.items() if features(c) is not None]) if chk else ((), ())
    scores = predict(np.array(rep["weights"]), np.array(X)) if X else np.array([])
    drop = sorted(i for i, s in zip(ids, scores) if s >= args.threshold)
    args.out_ids.parent.mkdir(parents=True, exist_ok=True)
    args.out_ids.write_text("".join(i + "\n" for i in drop))
    print(f"{len(chk)} checked rows, {len(chk) - len(ids)} unreadable (never listed); {len(drop)} score >= "
          f"{args.threshold} ({len(drop) / max(1, len(ids)):.1%}) -> {args.out_ids}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--check", type=Path)
    p.add_argument("--events-dir", type=Path, default=V5)
    p.add_argument("--out", type=Path)
    p.add_argument("--apply", type=Path, help="check rows to score with --weights (production)")
    p.add_argument("--weights", type=Path)
    p.add_argument("--threshold", type=float)
    p.add_argument("--out-ids", type=Path)
    args = p.parse_args()
    if args.apply:
        if not (args.weights and args.threshold is not None and args.out_ids):
            sys.exit("--apply needs --weights, --threshold and --out-ids")
        apply(args)
    elif args.manifest and args.check and args.out:
        calibrate(args)
    else:
        sys.exit("pass --manifest --check --out (calibration) or --apply ... (production)")


if __name__ == "__main__":
    main()
