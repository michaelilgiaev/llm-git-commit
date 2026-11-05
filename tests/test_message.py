"""Tests for message.extract_subject -- the commit-subject cleaner.

These tests pin the behaviour: last-non-empty-line extraction, fence/wrapper/period
stripping, and the chatter reject list. A regression here is exactly what would land
model chatter as a commit subject.
"""

from __future__ import annotations

import pytest

import message


def test_plain_subject_passes_through():
    assert message.extract_subject("Add greeting helper") == "Add greeting helper"


def test_takes_last_nonempty_line_not_first():
    # On a big diff the model emits a preamble; the real subject is the LAST line.
    reply = "Looking at the diff, the change adds a helper.\n\nAdd greeting helper"
    assert message.extract_subject(reply) == "Add greeting helper"


def test_blank_lines_are_ignored():
    assert message.extract_subject("\n\nFix typo\n\n") == "Fix typo"


def test_strips_wrapping_backticks():
    assert message.extract_subject("`Add parser flag`") == "Add parser flag"


def test_strips_wrapping_double_quotes():
    assert message.extract_subject('"Add parser flag"') == "Add parser flag"


def test_strips_trailing_period():
    assert message.extract_subject("Add parser flag.") == "Add parser flag"


def test_drops_closing_fence_line_and_uses_real_subject():
    # A fenced answer: the LAST non-empty line is the closing ``` -- it must be
    # dropped so the real subject (the line above) wins, not reduced to a backtick.
    reply = "```\nAdd greeting helper\n```"
    assert message.extract_subject(reply) == "Add greeting helper"


def test_language_tagged_fence_is_dropped():
    reply = "```text\nAdd greeting helper\n```"
    assert message.extract_subject(reply) == "Add greeting helper"


@pytest.mark.parametrize(
    "reply",
    [
        "Here is the commit subject:",      # trailing colon lead-in
        "Write the commit subject line",    # echoed instruction
        "Write a commit subject for this",  # echoed instruction
        "I will write the subject",          # first-person opener
        "Let me summarize the changes",      # lead-in
        "Based on the diff",                 # lead-in
        "Done. The commit subject is ready above.",  # acknowledgement chatter
        "Perfect, here you go",              # acknowledgement
        "- add a helper",                    # a single list item, not the change
        "* add a helper",                    # bullet
        "+ add a helper",                    # bullet
        "`",                                 # punctuation debris (lone fence)
        "-",                                 # punctuation debris
        "",                                  # empty
    ],
)
def test_rejects_non_subjects(reply):
    assert message.extract_subject(reply) == ""


def test_rejects_prose_far_past_cap():
    prose = "This change " + "word " * 40     # well over the 100-char cap
    assert message.extract_subject(prose) == ""


def test_real_subject_slightly_over_72_is_kept():
    # We reject only past 100, not 72 -- a real subject a little long is fine.
    subject = "Add a reasonably descriptive commit subject line that runs a little past seventy-two"
    assert 72 < len(subject) <= 100
    assert message.extract_subject(subject) == subject


def test_legit_subject_starting_with_add_is_untouched():
    assert message.extract_subject("Add -v flag to parser") == "Add -v flag to parser"


def test_short_this_line_is_a_subject():
    # "This adds ..." is a subject. Rejecting it used to throw the line away
    # and drop the user into the editor.
    assert message.extract_subject("This adds a compile script") == "This adds a compile script"
