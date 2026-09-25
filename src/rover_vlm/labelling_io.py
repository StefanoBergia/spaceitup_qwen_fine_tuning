"""Client-side plumbing shared by the Cosmos labelling scripts.

scripts/label_traces.py and scripts/label_instructions.py both talk to the endpoint that
slurm/serve_cosmos3.sbatch publishes, and both write a JSONL file that must survive being
killed and resumed. The rules for that file live here once:

* a non-empty output is never truncated unless --overwrite says so (a full run is hours);
* --resume keeps every finished row and drops error rows, so a failed request is retried
  rather than shadowed — a row that ran and merely failed a check is a result, not an error.
"""

import base64
import json
import os
from pathlib import Path


def resolve_endpoint(explicit, endpoint_file):
    if explicit:
        return explicit
    endpoint_file = Path(endpoint_file)
    if not endpoint_file.exists():
        raise SystemExit(
            f"No endpoint at {endpoint_file}.\n"
            "Start the server with 'sbatch slurm/serve_cosmos3.sbatch' and wait for it to\n"
            "report READY, or pass --endpoint http://<host>:<port>/v1 explicitly."
        )
    return endpoint_file.read_text().strip()


def prepare_output(out_path, *, resume, overwrite):
    """(ids already done, file mode) for a JSONL output about to be written."""
    out_path = Path(out_path)
    if not (out_path.exists() and out_path.stat().st_size):
        return set(), "w"
    if not resume:
        if overwrite:
            return set(), "w"
        raise SystemExit(f"{out_path} exists and is not empty. Pass --resume to continue it, "
                         "or --overwrite to replace it.")
    kept = [json.loads(line) for line in out_path.open() if line.strip()]
    failed = sum(1 for r in kept if r.get("error"))
    kept = [r for r in kept if not r.get("error")]
    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
    os.replace(tmp, out_path)          # atomic: never leaves a half-written file
    done = {r["id"] for r in kept}
    print(f"resume:   {len(done)} done in {out_path.name}"
          + (f", {failed} failed rows dropped for retry" if failed else ""))
    return done, ("a" if done else "w")


def connect(base_url, model=None):
    """(openai client, served model id); a dead endpoint exits with one readable line."""
    try:
        import openai
    except ImportError:
        raise SystemExit("openai not installed — run 'uv sync' (it is a project dependency).")
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url, timeout=600)
    try:
        return client, model or client.models.list().data[0].id
    except Exception as e:  # noqa: BLE001 — a dead endpoint should read as one line, not a stack
        raise SystemExit(
            f"Cannot reach the server at {base_url} ({type(e).__name__}: {e}).\n"
            "Check the serve job is still RUNNING (squeue) and that\n"
            "outputs/cosmos3_endpoint.txt points at the node it is actually on."
        )


def image_url(image):
    """A path becomes a file:// URI (the server reads NFS directly); bytes become a data URL."""
    if isinstance(image, (bytes, bytearray)):
        return "data:image/jpeg;base64," + base64.b64encode(bytes(image)).decode()
    return Path(image).resolve().as_uri()


def chat_with_image(client, model, image, prompt, *, max_tokens, seed):
    """One image+text request -> (raw reply, finish_reason). Raises on transport errors.

    `image` is a file path or JPEG bytes."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url(image)}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=max_tokens,
        seed=seed,
    )
    return resp.choices[0].message.content or "", resp.choices[0].finish_reason
