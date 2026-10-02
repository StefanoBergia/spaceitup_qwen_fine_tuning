# Final-review fixes (I1, I2, M2)
- I1: `naming_image` now pastes the 2px x px canvas centred onto a black 2px x 2px square (all callers). Spec A4 item 6 added. PROMPT_VERSION stays d3. Tests: square size, black bands, panel pixels in the middle band; enlarge-crop test crop shifted to the middle band.
- I2: sbatch header example is now `NAMER=gemma STAGE=full sbatch ...  # re-submit after TIMEOUT; never OVERWRITE=1 on a full run`; README says OVERWRITE=1 is for gate re-runs only (README hunk staged alone; user's framing row left unstaged).
- M2: `d1_title(desc)` in audit_instruction_labels.py; used for <title> and the payload `title`, which the template JS uses for document.title and h1 (the JS overrode the Python title). Test for default vs gemma stem. README audit example has the Gemma `--out outputs/instr_d1_audit_gemma.html`.
- Tests: `uv run pytest -q -x` -> 995 passed.
