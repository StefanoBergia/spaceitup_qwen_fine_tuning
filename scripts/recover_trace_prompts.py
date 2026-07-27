"""Recover the literal prompt text that produced each outputs/traces/*.jsonl run.

    uv run scripts/recover_trace_prompts.py --transcript ~/.claude/projects/<proj>/<id>.jsonl

src/rover_vlm/traces.py was first committed already at v6, so the repo has no record of
what v1-v5 actually said. The session that wrote them does: every Write, Edit and Bash
patch is in the transcript, in order. This replays them, snapshots the file at the moment
each labelling run was launched, and renders that snapshot's prompt for a real sample.

Two things make the output trustworthy rather than a guess:

  * The replay must reproduce the current src/rover_vlm/traces.py **byte for byte** after
    the last mutation. If it does not, the reconstruction is rejected and nothing is
    written — a partial replay would silently produce prompts that never existed.
  * Snapshots are keyed on when a *run* happened, not on PROMPT_VERSION. The constant
    lagged: the file jumped "v3" -> "v5" directly, and the v4 runs were tagged with the
    --prompt-version flag while the module still said v3. Anchoring on the run is what
    makes each prompt the one that actually produced that JSONL.

Writes scripts/trace_prompt_history.json, which scripts/visualize_traces.py reads. That
JSON is the durable artifact — this script only works while the transcript survives.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = "src/rover_vlm/traces.py"
OUT = REPO_ROOT / "scripts" / "trace_prompt_history.json"

SPLITS = {
    "choice": REPO_ROOT / "data" / "prepared_habitat_choice_v2" / "eval.json",
    "path": REPO_ROOT / "data" / "prepared_habitat_v2" / "eval.json",
}

def bash_replacements(command):
    """The (old, new) pairs a Bash heredoc applies to traces.py, in source order.

    Some revisions were made by piping a Python heredoc to the shell rather than by the
    Edit tool, so the op replay cannot see them. Rather than transcribe them by hand, the
    heredoc is parsed: a `s = s.replace(A, B)` counts only while `s` currently holds
    traces.py, since the same heredocs patch scripts/label_traces.py with the same
    variable name straight afterwards.
    """
    import ast

    body = re.search(r"<<\s*'?EOF'?\n(.*?)\nEOF", command, re.S)
    if not body:
        return []
    try:
        tree = ast.parse(body.group(1))
    except SyntaxError:
        return []

    paths, target, out = {}, None, []
    for node in tree.body:
        for stmt in (node.body if isinstance(node, ast.If) else [node]):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            name = getattr(stmt.targets[0], "id", None)
            v = stmt.value
            # X = pathlib.Path("...")  — remember which file the handle points at
            if (isinstance(v, ast.Call) and getattr(v.func, "attr", "") == "Path"
                    and v.args and isinstance(v.args[0], ast.Constant)):
                paths[name] = v.args[0].value
            # s = X.read_text()  — everything patched from here on belongs to that file
            elif isinstance(v, ast.Call) and getattr(v.func, "attr", "") == "read_text":
                target = paths.get(getattr(v.func.value, "id", None))
            # s = s.replace(old, new)
            elif (isinstance(v, ast.Call) and getattr(v.func, "attr", "") == "replace"
                  and len(v.args) == 2 and all(isinstance(a, ast.Constant) for a in v.args)):
                if target and target.endswith(TARGET):
                    out.append((v.args[0].value, v.args[1].value))
    return out


def transcript_events(path):
    """(line_no, kind, payload) for every mutation of traces.py and every labelling run."""
    events = []
    for ln, line in enumerate(path.open()):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (rec.get("message") or {}).get("content")
        for c in content if isinstance(content, list) else []:
            if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                continue
            inp = c.get("input") or {}
            if str(inp.get("file_path", "")).endswith(TARGET) and c["name"] in ("Write", "Edit"):
                events.append((ln, c["name"].lower(), inp))
            elif c["name"] == "Bash":
                cmd = inp.get("command", "")
                if "label_traces.py" in cmd and "--task" in cmd:
                    for m in re.finditer(r"label_traces\.py([^\n|&;]*)", " ".join(cmd.split())):
                        task = re.search(r"--task (\w+)", m.group(1))
                        if task:
                            pv = re.search(r"--prompt-version ([\w]+)", m.group(1))
                            events.append((ln, "run", {"task": task.group(1),
                                                       "tag": pv.group(1) if pv else None}))
                elif TARGET in cmd and "replace(" in cmd:
                    events.append((ln, "bash", inp))
    return events


def replay(events):
    """Apply every mutation in order, snapshotting the file state at each labelling run.

    Ordering matters and the byte-equality check cannot police it — the same set of edits
    reaches the same final file whenever they are applied. So each Bash event applies only
    the replacements parsed out of *that* command, never a greedy sweep: the constant went
    "v3" -> "v5" -> "v6" at two specific moments, and a run that happened between them must
    see the state of that moment.
    """
    state, snapshots = None, []
    for ln, kind, payload in events:
        if kind == "write":
            state = payload["content"]
        elif kind == "edit":
            old, new = payload["old_string"], payload["new_string"]
            if state is None or state.count(old) != 1:
                sys.exit(f"line {ln}: cannot apply edit ({0 if state is None else state.count(old)} matches)")
            state = state.replace(old, new, 1)
        elif kind == "bash":
            # `new not in state` is what makes a retry a no-op. Testing `old` alone is not
            # enough: these patches insert *around* their anchor, so `old` survives into
            # `new` and a re-application would duplicate the insertion. (One command here
            # did fail and get retried, so this path is exercised; the patch then lands at
            # the failed attempt's position rather than the retry's, which is
            # indistinguishable in the file and has no labelling run in between.)
            for old, new in bash_replacements(payload["command"]):
                if old in state and new not in state:
                    state = state.replace(old, new, 1)
        elif kind == "run" and state is not None:
            snapshots.append({"line": ln, "task": payload["task"], "tag": payload["tag"],
                              "source": state})
    return state, snapshots


def render(source, task, record, version):
    """Import a historical snapshot as a module and ask it for the prompt it would build."""
    import importlib.util

    pkg = REPO_ROOT / "src" / "rover_vlm"
    tmp = pkg / f"_hist_{version}.py"          # inside the package: the relative import works
    tmp.write_text(source)
    try:
        spec = importlib.util.spec_from_file_location(f"rover_vlm._hist_{version}", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        builder = getattr(mod, "build_prompt", None)
        if builder:
            return builder(task, record)
        return getattr(mod, f"{task}_trace_prompt")(record)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transcript", type=Path, required=True, help="the session .jsonl")
    p.add_argument("--traces-dir", type=Path, default=REPO_ROOT / "outputs" / "traces")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    events = transcript_events(args.transcript)
    runs = [e for e in events if e[1] == "run"]
    print(f"{len(events) - len(runs)} mutations, {len(runs)} labelling runs in {args.transcript.name}")

    final, snapshots = replay(events)
    live = (REPO_ROOT / TARGET).read_text()
    if final != live:
        sys.exit(f"REJECTED: replay produced {len(final)} chars, {TARGET} on disk has {len(live)}. "
                 "The reconstruction is incomplete — refusing to write prompts that may never "
                 "have existed.")
    print(f"replay reproduces {TARGET} byte for byte ({len(live)} chars) — snapshots are exact")

    # A run's output file is <task>_<tag or module default>.jsonl. Resolve the tag the same
    # way scripts/label_traces.py does, so a snapshot lands on the file it really produced.
    have = {p.stem for p in args.traces_dir.glob("*.jsonl")}
    out, seen = {}, set()
    for snap in snapshots:
        tag = snap["tag"] or re.search(r'^PROMPT_VERSION = "(\w+)"', snap["source"], re.M).group(1)
        key = f"{snap['task']}_{tag}"
        if key not in have:
            continue
        seen.add(key)
        record = {r["id"]: r for r in json.loads(SPLITS[snap["task"]].read_text())}
        sample = json.loads((args.traces_dir / f"{key}.jsonl").open().readline())["id"]
        # `source` is the durable part: a prompt is only ever rendered for one sample, so
        # anything wanting this version's prompt for a *different* frame — as the report's
        # per-frame view does — has to re-render it rather than reuse the string.
        # A later run with the same tag overwrote the earlier JSONL, so last write wins.
        out[key] = {"task": snap["task"], "version": tag, "line": snap["line"],
                    "sample": sample, "source": snap["source"],
                    "prompt": render(snap["source"], snap["task"], record[sample], tag)}
    missing = sorted(have - seen)
    if missing:
        print(f"  no snapshot for: {', '.join(missing)} — those runs predate the transcript")

    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}  ({len(out)} prompts)")
    for key, v in sorted(out.items()):
        print(f"  {key:14} {len(v['prompt']):5} chars   from transcript line {v['line']}")


if __name__ == "__main__":
    main()
