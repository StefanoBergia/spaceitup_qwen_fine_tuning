import json

import pytest

from rover_vlm.labelling_io import prepare_output


def test_prepare_output_new_file(tmp_path):
    assert prepare_output(tmp_path / "x.jsonl", resume=False, overwrite=False) == (set(), "w")


def test_prepare_output_refuses_to_clobber(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"id": "a"}\n')
    with pytest.raises(SystemExit):
        prepare_output(p, resume=False, overwrite=False)
    assert prepare_output(p, resume=False, overwrite=True) == (set(), "w")


def test_prepare_output_resume_retries_errors_only(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [{"id": "a", "passed": False}, {"id": "b", "error": "Timeout"}, {"id": "c", "passed": True}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    done, mode = prepare_output(p, resume=True, overwrite=False)
    assert done == {"a", "c"} and mode == "a"
    assert [json.loads(l)["id"] for l in p.read_text().splitlines()] == ["a", "c"]


from rover_vlm.labelling_io import image_url


def test_image_url_path_and_bytes(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff\xd8xx")
    assert image_url(p).startswith("file://")
    assert image_url(b"\xff\xd8xx") == "data:image/jpeg;base64,/9h4eA=="
