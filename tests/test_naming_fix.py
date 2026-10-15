"""Object naming fix (spec 2026-10-15): prompt n2, the naming sample, the variants job, the names audit, renaming."""
import json
from pathlib import Path

import pytest

from rover_vlm.descriptions import label_object, naming_prompt

REPO = Path(__file__).resolve().parent.parent


def _norm(p):
    return " ".join(p.split())


def _reply(payload):
    return f"<answer>{json.dumps(payload)}</answer>"


@pytest.mark.parametrize("need", ["opening", "wall", "object", "anything"])
def test_n1_is_the_default_and_unchanged(need):
    assert naming_prompt(need) == naming_prompt(need, "n1")


def test_n2_wording():
    p = _norm(naming_prompt("wall", "n2"))
    assert "Name the single thing that fills most of the outline" in p
    assert "Look at what is inside the outline, not at what is next to it or behind it." in p
    assert "If the outline is mostly part of the building, say which part" in p
    assert "Name furniture or an appliance only if the outline is clearly on it" in p
    assert "such as a sofa or a counter" not in p
    assert "something long that the rover drives alongside or stops in front of" in p
    assert '"wall" (part of the building: a wall, a column or pillar, a door, a partition, a railing)' in p
    assert "Never use the words left or right" in p and "not a small part on it" in p
    assert "The rover drives through it." in _norm(naming_prompt("opening", "n2"))
    assert "something the rover drives past" in _norm(naming_prompt("object", "n2"))
    assert "something the rover drives past" in _norm(naming_prompt("anything", "n2"))
    with pytest.raises(ValueError):
        naming_prompt("object", "n9")


def test_label_object_uses_and_records_the_prompt_version():
    seen = []
    obj = {"object_id": "f:b1", "kind_hint": "wall", "image": None, "polygon": [], "crop_box": None}

    def ask(img, prompt, seed):
        seen.append(prompt)
        return _reply({"name": "white wall", "kind": "wall"}), "stop"
    row = label_object(ask, obj, seed=0, image=b"x", name_prompt="n2")
    assert row["passed"] and row["name_prompt"] == "n2" and seen[0] == naming_prompt("wall", "n2")
    assert label_object(ask, obj, seed=0, image=b"x")["name_prompt"] == "n1"


def test_label_descriptions_has_the_flag():
    text = (REPO / "scripts" / "label_descriptions.py").read_text()
    assert '"--name-prompt"' in text and 'choices=["n1", "n2"]' in text
    assert "name_prompt=args.name_prompt" in text
