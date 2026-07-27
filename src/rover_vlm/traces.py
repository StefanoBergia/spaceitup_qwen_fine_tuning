"""Reasoning-trace labelling: build the prompts we send to Cosmos, read back what it says.

Pure and CPU-only — no network, no model. scripts/label_traces.py does the I/O.

The strategy is *justify the ground truth*: Cosmos is shown the frame together with the
known-correct answer and asked to explain why that answer is right. It never has to solve
the task, so a trace can never contradict its label.

That buys correctness at the cost of one specific hazard. The trace is destined for
Qwen3.5's `<think>` block, and at inference Qwen will NOT know the answer — so a trace
that says "since candidate 1 is given as correct" teaches exactly the wrong reflex. Every
such trace is worse than no trace at all. Two defences live here: the prompts forbid it,
and leakage_spans() detects it so a prompt revision can be judged on the numbers rather
than on vibes.
"""

import json
import re

from .habitat_choice import CANDIDATE_COLORS

# Bump when a prompt changes, so outputs/traces/<task>_<version>.jsonl from different
# revisions sit side by side and stay comparable on the same sample ids.
#
# v1 harvested Cosmos's <think> block and that was the wrong slot. Its <think> is private
# deliberation about *our request*, so every trace opened "Okay, let's see. The user wants
# me to explain why candidate 1 is..." and referred to "the answer", "the candidates
# listed" — meta-reasoning about a labelling task, not a rover reasoning about a room.
# v2 lets Cosmos think freely in <think> and throws it away, asking for the rover's
# reasoning as a *product* inside the answer JSON instead.
#
# v3 stopped making the path task echo its whole waypoint list (200 wasted tokens that
# pushed Cosmos into fenced JSON and lost the <answer> tags — it echoes just the goal now).
#
# v4 removed the concrete furniture example from the rules: 2 of 12 v2 traces parroted
# "the sofa and the kitchen counter" straight out of the prompt, inventing objects that
# were not in the frame. Illustrative nouns in a labelling prompt become hallucinations
# in the labels. This landed *after* the v3 run, so the v2 and v3 labels were both
# produced with the example still in place — the recovered prompt diffs say so even
# though this comment used to credit v3 with both changes.
#
# v5 bounded the scratch block. v4 told Cosmos to "think as long as you need", and on
# 2 of 20 choice samples it did exactly that — 3,300 words of deliberation, hitting the
# 4096-token cap before it ever closed </think>, so the sample yielded nothing. Healthy
# runs use 205-808 scratch words, so there was never anything to buy with more.
#
# v6 demands one JSON object in <answer>. Shortening the scratch made Cosmos skip the
# JSON envelope and write bare prose on 19 of 30 choice samples — the trace survived via
# salvage but the echoed answer, our only drift check, did not.
PROMPT_VERSION = "v6"

# Parallel to habitat_choice.CANDIDATE_COLORS — Cosmos sees colours, not RGB tuples, and
# naming them lets it refer to a candidate the way the image actually presents it.
CANDIDATE_COLOR_NAMES = ["orange", "sky blue", "bluish green", "vermillion", "reddish purple"]

# habitat_choice's per-candidate route shape, as something you can say in a sentence.
KIND_PHRASES = {
    "left": "veers to the left",
    "right": "veers to the right",
    "direct": "heads straight towards the goal",
}

SCENE_INTRO = (
    "This is the forward-facing camera view from a rover driving through an indoor scene."
)

# Repeated verbatim in both prompts: the whole point of the exercise.
#
# The framing rule matters more than the phrase blacklist. The trace becomes Qwen's inner
# monologue at inference, when it is looking at a room and has decided nothing yet — so it
# has to read as situated navigation reasoning, not as an explanation addressed to someone.
NO_LEAK_RULES = """Rules for `reasoning` — these matter more than the answer itself:
- Write it in the rover's own voice: first person, present tense, deciding right now from
  what the camera shows. "I can see...", "the floor ahead...", "that route would...".
- It must never betray that any answer was supplied. Do not mention a user, a question, a
  task, an explanation, a list of options given to you, or that some option is already
  known to be right. No "the answer is", "the correct one", "I need to reject", "as
  stated", "we are told". Reason towards the conclusion; never from it.
- Ground every claim in something actually visible. Name the furniture, wall, doorway or
  floor region you mean, and never name an object that is not in this image.
- Be specific rather than vague: identify which object, which wall, which opening. A
  sentence that would fit any indoor room is not worth writing.
- 60 to 120 words, one paragraph, no lists, no headings."""

# Cosmos always reasons before answering, and that reasoning always sounds like a model
# answering a request — so <think> is conceded to it as scratch space and discarded, and
# the deliverable is the answer's `reasoning` field, written as an artifact rather than as
# private deliberation. The brevity instruction is load-bearing, not politeness: told to
# think freely, it will spend 3,000+ words and never reach <answer>.
SCRATCH_NOTE = (
    "Keep your private working in <think> short — a few sentences at most. It is scratch "
    "and is thrown away, so it does not matter what it sounds like and there is nothing to "
    "gain by deliberating at length. Everything that matters goes in <answer>."
)


def _format_block(answer_shape):
    # The "one JSON object, nothing else" line is load-bearing. Without it Cosmos writes the
    # reasoning as bare prose inside <answer> roughly two thirds of the time: the trace is
    # still salvageable, but the echoed answer is gone and with it the only check that it
    # was looking at the right sample.
    return (
        f"{SCRATCH_NOTE}\n\n"
        "Reply in exactly this format and nothing else:\n"
        "<think>\n"
        "(your private working — discarded)\n"
        "</think>\n\n"
        "<answer>\n"
        f"{answer_shape}\n"
        "</answer>\n\n"
        "The <answer> block must hold exactly one JSON object and nothing else — no code "
        "fence, no sentences outside the braces. All of your prose belongs inside the "
        '"reasoning" string.'
    )


def choice_trace_prompt(record):
    """Prompt asking Cosmos to justify the correct candidate of a classification sample.

    Takes a record from data/prepared_habitat_choice*/ — the `choice_meta` sidecar carries
    the label, candidate count and route kinds that the answer string alone cannot express.
    """
    meta = record["choice_meta"]
    label, n = meta["label"], meta["n_candidates"]

    kinds = meta.get("kinds") or []
    legend = []
    for i in range(n):
        colour = CANDIDATE_COLOR_NAMES[i] if i < len(CANDIDATE_COLORS) else "a distinct colour"
        kind = KIND_PHRASES.get(kinds[i] if i < len(kinds) else None, "runs towards the goal")
        legend.append(f"- Candidate {i}: drawn in {colour}, badge {i}; it {kind}.")

    others = [str(i) for i in range(n) if i != label]
    return f"""{SCENE_INTRO}

{n} candidate paths to the goal have been drawn onto it. Each is a coloured polyline with a
numbered badge, and the goal itself is marked in white.

{chr(10).join(legend)}

Exactly one candidate is genuinely traversable. The others are not: they cut through a
wall, cross furniture, or leave the navigable floor. The traversable one is candidate
{label}.

Produce the rover's reasoning for settling on candidate {label}. It should say what makes
that route clear underfoot, and dismiss {"candidate " + others[0] if len(others) == 1 else "candidates " + ", ".join(others[:-1]) + " and " + others[-1]}, naming for each the
specific obstacle or surface it runs into.

{NO_LEAK_RULES}

{_format_block('{"reasoning": "<the rover\'s reasoning, 60-120 words>", "choice": %d}' % label)}"""


def path_summary(path, goal):
    """Derived facts about a ground-truth route, so the prompt can be specific about it.

    Returns direction ("left"/"right"/"straight"), waypoint counts, and where the route
    first disappears behind something — the occlusion story is the part of this task a
    trace most needs to explain.
    """
    hidden = [i for i, p in enumerate(path) if len(p) > 2 and not p[2]]
    dx = (goal[0] - path[0][0]) if path else 0.0
    return {
        "direction": "left" if dx < -0.08 else "right" if dx > 0.08 else "straight",
        "n_points": len(path),
        "n_hidden": len(hidden),
        "first_hidden": hidden[0] if hidden else None,
        "goal_visible": bool(len(goal) > 2 and goal[2]),
    }


def path_trace_prompt(record):
    """Prompt asking Cosmos to justify the ground-truth route of a regression sample.

    Regression records carry no metadata sidecar, so the facts are derived from the
    answer JSON in conversations[1] via path_summary().
    """
    gt = json.loads(record["conversations"][1]["value"])
    path, goal = gt["path"], gt["goal"]
    s = path_summary(path, goal)

    heading = {
        "left": "bends to the left of where the rover is pointing",
        "right": "bends to the right of where the rover is pointing",
        "straight": "runs essentially straight ahead",
    }[s["direction"]]

    if s["n_hidden"]:
        occlusion = (
            f"From waypoint {s['first_hidden']} onward the route passes behind something and is "
            f"no longer directly visible ({s['n_hidden']} of {s['n_points']} waypoints are "
            f"occluded). The goal itself is {'visible' if s['goal_visible'] else 'not visible'}."
        )
    else:
        occlusion = "The entire route stays on visible, unobstructed floor."

    return f"""{SCENE_INTRO}

The rover must reach a goal ahead of it. The route it should take, as normalized image
coordinates where x runs right and y runs down and both lie in [0,1], is:

{json.dumps(gt)}

Each waypoint is [x, y, v], where v is 1 when the point lies on visible unobstructed
ground and 0 when it is hidden behind an obstacle.

This route starts at the bottom of the frame, at the rover's own position, and {heading}.
{occlusion}

Produce the rover's reasoning for taking this route. It should cover three things and
nothing else: what makes the floor along it actually drivable, what obstacle or piece of
furniture forces it into the shape it has, and what it passes behind where it stops being
visible. Do not inventory the room — mention an object only when it explains the route.
Do not restate the coordinates.

{NO_LEAK_RULES}

{_format_block('{"reasoning": "<the rover\'s reasoning, 60-120 words>", "goal": '
               + json.dumps(goal) + '}')}"""


# --- reading Cosmos back -------------------------------------------------------------

_THINK_OPEN = re.compile(r"<think>", re.I)
_THINK = re.compile(r"<think>(.*?)</think>", re.I | re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def answer_payload(answer):
    """The answer block as a dict, tolerating prose wrapped around the JSON. None if absent.

    None means "could not verify", which is not the same as "wrong" — the caller should
    keep those apart, or a formatting slip gets scored as the model drifting off-sample.
    """
    if not answer:
        return None
    embedded = _JSON_OBJ.search(answer)
    for candidate in (answer, embedded.group(0) if embedded else None):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def trace_from_answer(answer):
    """The deliverable: the rover's reasoning, or None.

    Since v2 this — not the <think> block — is the trace. Cosmos's <think> is its private
    working about our request and always reads like it; see PROMPT_VERSION.

    Cosmos sometimes ignores the JSON envelope and writes the reasoning as bare prose
    (1 in 5 on the first v2 run). That prose is perfectly good, so it is salvaged rather
    than discarded — at full-dataset scale, dropping every formatting slip would quietly
    throw away a fifth of the labels.
    """
    if not answer:
        return None
    payload = answer_payload(answer)
    if payload is not None:
        reasoning = payload.get("reasoning")
        return reasoning.strip() or None if isinstance(reasoning, str) else None
    return answer.strip() or None


def parse_trace(text):
    """Cosmos output -> (scratchpad, answer). Either may be None; nothing raises.

    The first element is the <think> block, which since v2 is kept only for debugging —
    trace_from_answer() extracts the trace we actually keep. Truncation is the expected
    failure (reasoning models run long), so an unterminated `<think>` still yields
    whatever arrived rather than discarding the sample.
    """
    text = text or ""
    m = _ANSWER.search(text)
    if m:
        answer = m.group(1).strip()
    else:
        # Cosmos regularly drops the <answer> tags and emits the JSON on its own, often
        # pretty-printed inside a ``` fence. The content is fine, so take the JSON object
        # that follows the think block rather than throwing the sample away. Searching
        # only after </think> keeps braces inside the scratch from being mistaken for it.
        closing = re.search(r"</think>", text, re.I)
        tail = text[closing.end():] if closing else text
        embedded = _JSON_OBJ.search(tail)
        answer = embedded.group(0).strip() if embedded else None

    m = _THINK.search(text)
    if m:
        return m.group(1).strip() or None, answer

    m = _THINK_OPEN.search(text)
    if m:  # opened but never closed — generation hit the token limit mid-thought
        tail = text[m.end():]
        tail = tail[: tail.index("<answer>")] if "<answer>" in tail else tail
        return tail.strip() or None, answer

    # No think tags at all: take everything before <answer> as the reasoning.
    body = text[: text.index("<answer>")] if "<answer>" in text else text
    return body.strip() or None, answer


# Phrasings that betray the model was handed the answer. Deliberately narrow: these must
# fire on genuine leaks and stay silent on legitimate reasoning, since the rate they
# produce is what we steer prompt revisions by.
_LEAK_PATTERNS = [
    r"\b(?:i am|i'm|we are|we're|you are) told\b",
    r"\byou (?:said|told me|mentioned|indicated|specified|provided|gave)\b",
    r"\bas (?:you |the prompt |the question )?(?:stated|mentioned|indicated|specified|noted)\b",
    r"\bground[\s-]?truth\b",
    r"\bthe (?:given|provided|stated|specified|supplied|known|correct|expected) "
    r"(?:answer|label|choice|solution|route|path)\b",
    r"\baccording to the (?:prompt|question|instruction|metadata|information|description)\b",
    r"\bsince (?:it is |it's |we are |i am |this is )?(?:given|stated|specified|supplied)\b",
    r"\bthe (?:prompt|question|instructions?|task) (?:says?|states?|tells?|indicates?)\b",
    r"\bbased on the (?:given|provided|supplied) (?:information|answer|label|data|metadata)\b",
    r"\bwhich (?:is|was) (?:given|provided|stated|specified) (?:as|to be)\b",
    r"\b(?:is|was|has been) (?:given|provided|identified|designated) as (?:the )?"
    r"(?:correct|traversable|right|answer)\b",
    r"\bmetadata\b",
    # --- added after the v1 smoke run, which produced leaks these missed ---
    # "The answer says candidate 1 is the correct one" slipped through every pattern above.
    r"\bthe answer\b",
    r"\bthe correct (?:one|candidate|option|choice|path|route)\b",
    r"\bi need to (?:reject|explain|justify|show|prove)\b",
    r"\b(?:the|a) user\b",
    r"\bcandidate \d+ is (?:the )?correct\b",
    r"\brecall the candidates\b",
    # Meta-framing: the trace is the rover's monologue, so it must not narrate producing it.
    r"\b(?:let me|i(?:'| a)?ll) (?:explain|justify|describe why)\b",
    # --- added after the full path run (job 88327), which the v1-era patterns missed on
    # ~5% of path traces. The path prompt hands over a route ("the route it should take is
    # {json}"), so the model refers back to "the planned path" / "the pre-planned
    # trajectory" — deferring to a route that will not exist at inference. This is the
    # handed-route family; the choice task, which hands no route, never produces it. The
    # boundary is deliberately CLEAR-ONLY: bare "intended path" is left out, because it
    # reads as the rover's own intent rather than a plan it was given.
    r"\b(?:the|this|that|a|my|entire|its) (?:pre-?planned|planned|designated|prescribed|"
    r"assigned|predetermined|required|mapped|charted|plotted|computed|calculated|known|"
    r"given|provided|specified|correct|optimal|reference|suggested|recommended) "
    r"(?:path|route|trajectory|course|line|way)\b",
    r"\b(?:follow|following|trust|adhere to|stick to) the "
    r"(?:pre-?planned|planned|designated|prescribed|given|known)\b",
    # Two v1-era patterns were too broad and mis-fired on the full run. Bare "explains why"
    # hit ordinary physical causation ("the cabinet explains why the goal is hidden"); the
    # meta-framing form it was meant for is already covered above. A bare
    # question|prompt|task|instruction hit "complete my task" / "this task" in rover voice.
    # Both are re-added here in anchored forms that keep the giveaway sense only.
    r"\bthe (?:question|prompt|instructions?)\b",
    r"\bthe task (?:is|was|says?|states?|asks?|given|requires?)\b",
]
_LEAK = re.compile("|".join(_LEAK_PATTERNS), re.I)


def leakage_spans(trace):
    """Substrings that reveal the answer was handed over. Empty list is what we want."""
    return [m.group(0) for m in _LEAK.finditer(trace or "")]


def answer_matches(task, answer, record):
    """Did Cosmos echo back the very answer it was asked to justify?

    It is never asked to *solve* anything, so this is not an accuracy metric — it is a
    drift check. A mismatch means the reply wandered off the sample it was shown, which
    makes the accompanying trace untrustworthy even though the label is still correct.
    """
    parsed = answer_payload(answer)
    if parsed is None:
        return False
    try:
        truth = json.loads(record["conversations"][1]["value"])
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        return False
    if task == "choice":
        return parsed.get("choice") == truth.get("choice")
    # Path echoes only the goal: re-emitting the whole waypoint list cost ~200 tokens,
    # pushed Cosmos into pretty-printed fenced JSON (which lost the <answer> tags), and
    # verified nothing extra. Three numbers catch the same drift.
    return parsed.get("goal") == truth.get("goal")


def build_prompt(task, record):
    """Dispatch on task name, matching the --task flag of scripts/label_traces.py."""
    if task == "choice":
        return choice_trace_prompt(record)
    if task == "path":
        return path_trace_prompt(record)
    raise ValueError(f"unknown task {task!r} (expected 'choice' or 'path')")


__all__ = [
    "CANDIDATE_COLOR_NAMES",
    "KIND_PHRASES",
    "PROMPT_VERSION",
    "answer_matches",
    "answer_payload",
    "build_prompt",
    "choice_trace_prompt",
    "leakage_spans",
    "parse_trace",
    "trace_from_answer",
    "path_summary",
    "path_trace_prompt",
]
