import json
import re

import pytest

from rover_vlm.traces import (
    answer_matches,
    build_prompt,
    choice_trace_prompt,
    leakage_spans,
    parse_trace,
    path_summary,
    path_trace_prompt,
    trace_from_answer,
)


def _choice_record(label=1, n=3, kinds=("right", "right", "direct")):
    return {
        "id": "scene_c000",
        "image": ["/nfs/x.jpg"],
        "conversations": [{"from": "human", "value": "<image>\n..."},
                          {"from": "gpt", "value": '{"choice":%d}' % label}],
        "choice_meta": {"label": label, "accepted": [label], "n_candidates": n,
                        "kinds": list(kinds), "margin": 0.33, "near_symmetric": False},
    }


def _path_record(path, goal):
    gt = {"path": path, "goal": goal}
    return {
        "id": "scene_c000",
        "image": ["/nfs/x.png"],
        "conversations": [{"from": "human", "value": "<image>\n..."},
                          {"from": "gpt", "value": json.dumps(gt)}],
    }


# --- choice prompt ------------------------------------------------------------------


def test_choice_prompt_legends_every_candidate_and_names_the_label():
    p = choice_trace_prompt(_choice_record(label=1, n=3))
    for i in range(3):
        assert f"Candidate {i}:" in p
    assert "Candidate 3:" not in p
    assert "candidate 1" in p
    assert '"choice": 1' in p


def test_prompts_ask_for_reasoning_inside_the_answer_not_the_think_block():
    """v2's whole point: <think> is Cosmos's scratch, the trace is a field of <answer>."""
    for p in (choice_trace_prompt(_choice_record()),
              path_trace_prompt(_path_record([[0.5, 1.0, 1]], [0.5, 0.7, 1]))):
        assert '"reasoning"' in p
        assert "discarded" in p


def test_choice_prompt_rejects_the_other_candidates_by_number():
    kinds = ("right", "left", "direct", "left", "right")
    p = choice_trace_prompt(_choice_record(label=2, n=5, kinds=kinds))
    assert "candidates 0, 1, 3 and 4" in p


def test_choice_prompt_tolerates_kinds_shorter_than_n_candidates():
    p = choice_trace_prompt(_choice_record(label=0, n=5, kinds=("right", "left")))
    assert "Candidate 4:" in p


def test_choice_prompt_handles_a_single_other_candidate():
    p = choice_trace_prompt(_choice_record(label=0, n=2, kinds=("direct", "left")))
    assert "candidate 1" in p
    assert "candidates" not in p.split("reject")[1].split("\n")[0]


def test_choice_prompt_survives_more_candidates_than_named_colours():
    rec = _choice_record(label=0, n=7, kinds=("direct",) * 7)
    p = choice_trace_prompt(rec)
    assert "Candidate 6:" in p
    assert "a distinct colour" in p


def test_choice_prompt_tolerates_missing_kinds():
    rec = _choice_record(label=0, n=3)
    rec["choice_meta"]["kinds"] = None
    assert "Candidate 2:" in choice_trace_prompt(rec)


# --- path summary + prompt ----------------------------------------------------------


def test_path_summary_direction_and_occlusion():
    path = [[0.8, 1.0, 1], [0.7, 0.8, 1], [0.6, 0.7, 0], [0.5, 0.6, 0]]
    s = path_summary(path, [0.5, 0.6, 0])
    assert s["direction"] == "left"
    assert (s["n_points"], s["n_hidden"], s["first_hidden"]) == (4, 2, 2)
    assert s["goal_visible"] is False


def test_path_summary_straight_when_lateral_shift_is_small():
    path = [[0.5, 1.0, 1], [0.5, 0.8, 1]]
    assert path_summary(path, [0.52, 0.7, 1])["direction"] == "straight"


def test_path_summary_right_and_fully_visible():
    path = [[0.2, 1.0, 1], [0.4, 0.8, 1]]
    s = path_summary(path, [0.6, 0.7, 1])
    assert s["direction"] == "right"
    assert s["n_hidden"] == 0 and s["first_hidden"] is None and s["goal_visible"] is True


def test_path_prompt_describes_occlusion_when_present():
    rec = _path_record([[0.8, 1.0, 1], [0.6, 0.8, 0]], [0.5, 0.7, 0])
    p = path_trace_prompt(rec)
    assert "From waypoint 1 onward" in p
    assert "not visible" in p


def test_path_prompt_says_so_when_nothing_is_occluded():
    rec = _path_record([[0.5, 1.0, 1], [0.5, 0.8, 1]], [0.5, 0.7, 1])
    p = path_trace_prompt(rec)
    assert "entire route stays on visible" in p
    assert "From waypoint" not in p


# --- parsing ------------------------------------------------------------------------


def test_parse_trace_wellformed():
    scratch, answer = parse_trace('<think>\nThe sofa blocks it.\n</think>\n\n<answer>\n{"choice": 1}\n</answer>')
    assert scratch == "The sofa blocks it."
    assert answer == '{"choice": 1}'


# --- the trace itself now comes out of the answer JSON ------------------------------


def test_trace_from_answer_extracts_the_reasoning_field():
    raw = ('<think>Okay, the user wants me to justify candidate 1.</think>\n'
           '<answer>{"reasoning": "The tiled strip past the sofa stays open.", "choice": 1}</answer>')
    scratch, answer = parse_trace(raw)
    assert "the user wants me" in scratch          # Cosmos's meta-chatter stays quarantined
    assert trace_from_answer(answer) == "The tiled strip past the sofa stays open."


@pytest.mark.parametrize("answer", [
    None, "", "   ", '{"choice": 1}', '{"reasoning": null, "choice": 1}',
    '{"reasoning": "   ", "choice": 1}', '{"reasoning": 42}',
])
def test_trace_from_answer_returns_none_when_there_is_no_usable_reasoning(answer):
    assert trace_from_answer(answer) is None


def test_trace_from_answer_salvages_bare_prose():
    """Cosmos dropped the JSON envelope on 1 of 5 v2 samples but the prose was fine.

    Discarding those would silently lose ~20% of the labels at full-dataset scale.
    """
    prose = "I can see the floor between the conference table and the counter is open."
    assert trace_from_answer(prose) == prose


def test_trace_from_answer_finds_json_embedded_in_prose():
    raw = 'Here you go:\n{"reasoning": "The aisle past the bench is clear.", "choice": 2}\nDone.'
    assert trace_from_answer(raw) == "The aisle past the bench is clear."


def test_answer_payload_distinguishes_unverifiable_from_wrong():
    """A formatting slip must not be scored as the model drifting off-sample."""
    from rover_vlm.traces import answer_payload
    assert answer_payload("bare prose, no json") is None          # unverifiable
    assert answer_payload('{"choice": 0}') == {"choice": 0}        # verifiable, just wrong


def test_parse_trace_missing_answer():
    trace, answer = parse_trace("<think>The sofa blocks it.</think>")
    assert trace == "The sofa blocks it."
    assert answer is None


def test_parse_trace_unterminated_think_keeps_partial_reasoning():
    trace, answer = parse_trace("<think>\nThe sofa blocks it and then")
    assert trace == "The sofa blocks it and then"
    assert answer is None


def test_parse_trace_without_think_tags():
    trace, answer = parse_trace('Reasoning here.\n<answer>{"choice": 0}</answer>')
    assert trace == "Reasoning here."
    assert answer == '{"choice": 0}'


@pytest.mark.parametrize("text", ["", None, "   "])
def test_parse_trace_empty_inputs(text):
    assert parse_trace(text) == (None, None)


# --- leakage ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    "Since candidate 1 is given as the correct answer, the sofa must be clear.",
    "We are told the route bends left around the table.",
    "As stated, path 2 crosses the wall.",
    "The ground truth path avoids the counter.",
    "According to the prompt, candidate 0 is traversable.",
    "You said the goal is behind the doorway.",
    "Based on the provided information, the floor is open.",
    "The metadata lists three candidates.",
    "Candidate 1 is designated as correct because the aisle is clear.",
    # --- verbatim from the v1 smoke run; every one of these got past the first pattern set
    "The answer says candidate 1 is the correct one.",
    "The correct answer is 4, so I need to reject the others.",
    "Okay, let's see. The user wants me to explain why candidate 4 is traversable.",
    "First, I need to recall the candidates.",
    "Let me explain why the sofa forces a detour.",
])
def test_leakage_spans_flags_leaks(bad):
    assert leakage_spans(bad), f"should have flagged: {bad}"


@pytest.mark.parametrize("clean", [
    "Candidate 1 follows the open tiled floor between the sofa and the kitchen counter, "
    "while candidate 0 climbs straight over the armchair and candidate 2 ends inside the wall.",
    "The route bends right to clear the dining table, then disappears behind the partition "
    "where the corridor turns.",
    "The correct thing to do is hug the left wall, since the rug is raised.",
    "",
    # Rover-voice prose in the shape v2 asks for — the detector widened a lot after the
    # v1 run, so these guard the widening against firing on legitimate traces.
    "I can see open tiled floor running from my wheels to the doorway, with the armchair "
    "well clear to my right. Path 3 would put me into the dining table's near leg, and "
    "path 0 climbs the step up to the kitchen, so I take path 1 along the tiles.",
    "The rug ahead is flat and its edge is not raised, so I can cross it directly rather "
    "than skirting left around the coffee table.",
    "My route bends right to miss the counter overhang, then drops out of sight behind "
    "the partition wall before reaching the goal.",
])
def test_leakage_spans_silent_on_clean_traces(clean):
    assert leakage_spans(clean) == []


def test_leakage_spans_handles_none():
    assert leakage_spans(None) == []


# --- answer echo (drift check) ------------------------------------------------------


def test_answer_matches_choice():
    rec = _choice_record(label=1)
    assert answer_matches("choice", '{"choice": 1}', rec) is True
    assert answer_matches("choice", '{"choice": 0}', rec) is False


def test_answer_matches_path_checks_the_goal_echo():
    """Path echoes only the goal — the full waypoint list cost tokens and verified nothing."""
    path, goal = [[0.5, 1.0, 1], [0.5, 0.8, 0]], [0.5, 0.7, 0]
    rec = _path_record(path, goal)
    assert answer_matches("path", json.dumps({"reasoning": "x", "goal": goal}), rec) is True
    assert answer_matches("path", json.dumps({"reasoning": "x", "goal": [0.1, 0.2, 1]}), rec) is False


def test_path_prompt_does_not_make_the_model_echo_the_waypoints():
    rec = _path_record([[0.834, 1.0, 1], [0.5, 0.6, 0]], [0.5, 0.6, 0])
    p = path_trace_prompt(rec)
    # "<answer>" is also mentioned in prose, so match the real block rather than index into splits
    answer_block = re.search(r"<answer>(.*?)</answer>", p, re.S).group(1)
    assert "0.834" not in answer_block          # the path must not be re-emitted
    assert '"goal"' in answer_block             # but the cheap drift check survives


def test_parse_trace_recovers_json_when_answer_tags_are_missing():
    """Cosmos often drops <answer> and emits fenced JSON instead; that is still usable."""
    raw = ('<think>working...</think>\n```json\n'
           '{\n  "reasoning": "The tiled aisle stays clear.",\n  "goal": [0.5, 0.6, 0]\n}\n```')
    _, answer = parse_trace(raw)
    assert trace_from_answer(answer) == "The tiled aisle stays clear."


def test_parse_trace_ignores_braces_inside_the_think_block():
    raw = '<think>maybe {"choice": 9} is right</think>\n{"reasoning": "Clear.", "choice": 1}'
    _, answer = parse_trace(raw)
    assert json.loads(answer)["choice"] == 1


@pytest.mark.parametrize("answer", [None, "", "not json", "[1, 2]", '"a string"', "{}"])
def test_answer_matches_rejects_unusable_answers(answer):
    assert answer_matches("choice", answer, _choice_record(label=1)) is False


def test_answer_matches_survives_a_malformed_record():
    assert answer_matches("choice", '{"choice": 1}', {"conversations": []}) is False


# --- dispatch -----------------------------------------------------------------------


def test_build_prompt_dispatches():
    assert build_prompt("choice", _choice_record()) == choice_trace_prompt(_choice_record())
    rec = _path_record([[0.5, 1.0, 1]], [0.5, 0.7, 1])
    assert build_prompt("path", rec) == path_trace_prompt(rec)


def test_build_prompt_rejects_unknown_task():
    with pytest.raises(ValueError, match="unknown task"):
        build_prompt("nope", _choice_record())


# --- the prompts must not themselves leak -------------------------------------------


@pytest.mark.parametrize("noun", ["kitchen counter", "sofa", "armchair", "coffee table"])
def test_prompts_name_no_concrete_furniture(noun):
    """Illustrative nouns get parroted into traces as objects that are not in the frame.

    2 of 12 v2 path traces claimed a "kitchen counter" purely because the prompt's example
    mentioned one. A labelling prompt must not supply vocabulary for the scene.
    """
    for p in (choice_trace_prompt(_choice_record()),
              path_trace_prompt(_path_record([[0.5, 1.0, 1]], [0.5, 0.7, 1]))):
        assert noun not in p.lower()


def test_prompts_never_contain_an_image_token():
    """The <image> sentinel is Qwen's training format; the OpenAI API sends a content part."""
    assert "<image>" not in choice_trace_prompt(_choice_record())
    assert "<image>" not in path_trace_prompt(_path_record([[0.5, 1.0, 1]], [0.5, 0.7, 1]))
