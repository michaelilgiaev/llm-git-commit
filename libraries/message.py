"""message.py - turn a model reply into ONE clean commit subject line.

A small local model is pinned by a system prompt to emit ONE subject line, but on a
LARGE diff it can still prepend a short reasoning preamble with the real subject as
the LAST line, or fence its answer. This cleaner is readable and unit-tested. So we:
    * take the LAST non-empty line (not the first),
    * drop fence lines first,
    * strip wrapping backticks/quotes and a trailing period,
    * reject obvious non-subjects (chatter, list items, lead-ins, prose).
An empty result means "this reply is not a subject"; the commit path then
uses the harness draft.
"""

from __future__ import annotations

import re

# A fence line: ``` optionally followed by a language tag, nothing else.
_FENCE_RE = re.compile(r"^```[A-Za-z0-9]*$")

# Reject shapes that are NOT a subject:
#   * a reasoning / first-person opener,
#   * an acknowledgement (what a model emits when something made it keep talking),
# all anchored at the START so a legitimate subject ("This fixes ...") is only
# caught when it literally opens with one of these lead-ins. A prefix matches a
# whole word: "Sure" rejects "Sure thing" and keeps "Surely" / "Greatly".
_LEADINS = (
    "Looking at ", "Here ", "I ", "Let me ", "Sure", "Okay", "Based on ",
    "The staged ",
    # The instruction, if a model echoes it instead of writing a subject.
    "Write the commit subject", "Write a commit subject",
)
_ACKS = (
    "Done", "Perfect", "Got it", "Great", "Understood", "Certainly", "Of course",
)
# A harness draft that is only one of these words carries no change.
_BARE_ACK = re.compile(
    r"^(?:done|perfect|got it|great|understood|certainly|of course|sure|okay|ok|yes)$",
    re.I,
)

# One line of "; "-separated clauses. A single change can be short. A pile of
# clauses is clipped at a clause boundary. An unsplittable ramble past the cap
# is rejected (it is prose, not a commit line).
_MAX_LEN = 360


def _strip_wrappers(line: str) -> str:
    """Strip a single pair of wrapping backticks or double-quotes, then a trailing
    period -- the three cosmetic things a model adds around an otherwise clean line."""
    for q in ("`", '"'):
        if len(line) >= 2 and line.startswith(q) and line.endswith(q):
            line = line[1:-1]
    return line[:-1] if line.endswith(".") else line


def _fit(line: str) -> str:
    """Keep the line when it fits. When it is a run of clauses, keep whole
    clauses that fit. A single clause past the cap is prose and returns ''."""
    if len(line) <= _MAX_LEN:
        return line
    kept = []
    size = 0
    for part in (p.strip() for p in line.split(";")):
        if not part:
            continue
        extra = len(part) if not kept else len(part) + 2
        if size + extra > _MAX_LEN:
            break
        kept.append(part)
        size += extra
    if not kept:
        return ""
    return "; ".join(kept)


def _opens_with(line: str, prefix: str) -> bool:
    """True when `line` opens with `prefix` as a whole word.

    "I " and "Here " already end in a space, so "Ignore" and "Hereafter" stay.
    "Sure" rejects "Sure." and keeps "Surely".
    """
    if not line.startswith(prefix):
        return False
    if prefix.endswith(" "):
        return True
    rest = line[len(prefix):]
    return rest == "" or not rest[0].isalpha()


def _is_rejected(line: str) -> bool:
    """True when `line` is chatter/prose/a list item rather than a subject."""
    if line.endswith(":"):                      # lead-in to a list that followed
        return True
    if any(_opens_with(line, prefix) for prefix in _LEADINS + _ACKS):
        return True
    if line[:2] in ("- ", "* ", "+ "):          # a single bullet, not the whole change
        return True
    if not re.search(r"[A-Za-z0-9]", line):     # punctuation debris (lone fence/dash)
        return True
    return False


def extract_subject(reply: str) -> str:
    """Return a clean commit subject from the model's raw `reply`, or '' if none.

    Takes the LAST non-empty, non-fence line (both ends trimmed), strips wrappers
    and a trailing period, and returns '' when the result is rejected. The commit
    path then uses the harness draft.
    """
    candidate = ""
    for raw in reply.splitlines():
        line = raw.strip()
        if not line or _FENCE_RE.match(line):
            continue
        candidate = line                        # keep the last survivor
    if not candidate:
        return ""
    candidate = _fit(_strip_wrappers(candidate))
    return "" if (not candidate or _is_rejected(candidate)) else candidate


def keep_line(line: str) -> str:
    """Keep a harness draft the chatter filter dropped, when it still says something.

    First-person and ack-shaped sentences ("I added a check…", "Done waiting…",
    "Surely…") are still the line the program wrote. A bare "Done" is not.
    """
    text = " ".join((line or "").split()).strip()
    text = _strip_wrappers(text).strip()
    if text[:2] in ("- ", "* ", "+ "):
        text = text[2:].strip()
        if text[:1].isalpha():
            text = text[0].upper() + text[1:]
    if text.endswith((".", ":", "!")):
        text = text[:-1].strip()
    if not text or _BARE_ACK.match(text) or not re.search(r"[A-Za-z0-9]", text):
        return ""
    fitted = _fit(text)
    if fitted:
        return fitted
    clipped = text[:_MAX_LEN].rsplit(" ", 1)[0].strip()
    return clipped or text[:_MAX_LEN]
