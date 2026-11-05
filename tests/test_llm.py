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
    assert "the diff" in kw["messages"][1]["content"]
    assert llm.USER_PREAMBLE in kw["messages"][1]["content"]


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


def test_model_path_uses_configuration(monkeypatch):
    # huggingface_hub is a runtime-only dep (not in the test venv); inject a fake
    # module so _model_path's deferred `from huggingface_hub import hf_hub_download`
    # resolves to our capture without the real package installed.
    import sys
    import types

    captured = {}

    def _fake_download(*, repo_id, filename):
        captured.update(repo_id=repo_id, filename=filename)
        return "/tmp/fake.gguf"

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = _fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setenv("LLM_GIT_COMMIT_REPO", "me/r")
    monkeypatch.setenv("LLM_GIT_COMMIT_FILE", "m.gguf")

    assert llm._model_path() == "/tmp/fake.gguf"
    assert captured == {"repo_id": "me/r", "filename": "m.gguf"}
