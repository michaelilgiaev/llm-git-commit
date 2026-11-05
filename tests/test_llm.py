"""Tests for llm.py -- the local-model streaming loop, WITHOUT loading a real model.

We never load a GGUF (too heavy for a unit test and would hit the network on first
run). Instead we fake llm._load() to return a stub whose create_chat_completion
yields canned streaming chunks shaped exactly like llama-cpp-python's, and assert
the subject is assembled, the progress bar is driven, and failures return ''.
"""

from __future__ import annotations

import llm


def _chunks(pieces, finish="stop"):
    """Yield llama-cpp-style streaming chunks: a role-only opener, one chunk per
    text piece, then a final finish_reason chunk (as the real API does)."""
    yield {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
    for p in pieces:
        yield {"choices": [{"delta": {"content": p}, "finish_reason": None}]}
    yield {"choices": [{"delta": {}, "finish_reason": finish}]}


class _StubLlama:
    def __init__(self, pieces):
        self._pieces = pieces
        self.calls = []

    def create_chat_completion(self, **kw):
        self.calls.append(kw)
        return _chunks(self._pieces)


def test_assembles_streamed_subject_and_sends_prompt(monkeypatch):
    stub = _StubLlama(["Add ", "greeting ", "helper"])
    monkeypatch.setattr(llm, "_load", lambda: stub)

    out = llm.generate_subject("the diff")
    assert out == "Add greeting helper"

    kw = stub.calls[0]
    assert kw["stream"] is True
    assert kw["stop"] == ["\n"]
    assert kw["messages"][0]["content"] == llm.SYSTEM_PROMPT
    # User turn is the diff alone. A preamble in this turn is what the small
    # model used to echo back as the subject.
    assert kw["messages"][1]["content"] == "the diff"
    assert kw["temperature"] == 0


def test_progress_bar_is_driven(monkeypatch):
    monkeypatch.setattr(llm, "_load", lambda: _StubLlama(["a", "b", "c"]))
    seen = []
    monkeypatch.setattr(llm.progress, "bar",
                        lambda cur, total, **k: seen.append((cur, total)))
    done = {"n": 0}
    monkeypatch.setattr(llm.progress, "done", lambda: done.update(n=done["n"] + 1))

    llm.generate_subject("d")
    assert seen[0][0] == 0                     # starts at 0
    assert [c for c, _ in seen] == [0, 1, 2, 3]  # one tick per content chunk
    assert all(total == llm._MAX_TOKENS for _, total in seen)
    assert done["n"] == 1                       # bar cleared exactly once


def test_model_failure_returns_empty(monkeypatch):
    class _Boom:
        def create_chat_completion(self, **kw):
            raise RuntimeError("llama exploded")

    monkeypatch.setattr(llm, "_load", lambda: _Boom())
    assert llm.generate_subject("d") == ""


def test_system_prompt_stays_short():
    # A long prompt made the 1.5B model echo the instruction. Keep it one line.
    assert len(llm.SYSTEM_PROMPT) < 200
    assert "72" in llm.SYSTEM_PROMPT
    assert "scope number" in llm.SYSTEM_PROMPT


def test_model_path_uses_a_local_file_and_does_not_download(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(llm, "_MIN_BYTES", 1)
    monkeypatch.setattr(llm, "_local_file", lambda: gguf)
    monkeypatch.setattr(llm, "_cached_hf", lambda: None)

    def _boom(dest):
        raise AssertionError("downloaded")

    monkeypatch.setattr(llm, "_download", _boom)
    assert llm._model_path() == str(gguf)


def test_download_url_follows_configuration(monkeypatch):
    monkeypatch.setenv("LLM_GIT_COMMIT_REPO", "me/r")
    monkeypatch.setenv("LLM_GIT_COMMIT_FILE", "m.gguf")
    assert llm._download_url() == "https://huggingface.co/me/r/resolve/main/m.gguf"
