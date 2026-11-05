"""Tests for configuration -- the local GGUF repo/file, with env overrides.
There is no saved config and no API key any more."""

from __future__ import annotations

import configuration


def test_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("LLM_GIT_COMMIT_REPO", raising=False)
    monkeypatch.delenv("LLM_GIT_COMMIT_FILE", raising=False)
    assert configuration.model_repo() == configuration.DEFAULT_REPO
    assert configuration.model_file() == configuration.DEFAULT_FILE


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_GIT_COMMIT_REPO", "me/other-GGUF")
    monkeypatch.setenv("LLM_GIT_COMMIT_FILE", "other.gguf")
    assert configuration.model_repo() == "me/other-GGUF"
    assert configuration.model_file() == "other.gguf"


def test_blank_env_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_GIT_COMMIT_REPO", "   ")
    monkeypatch.setenv("LLM_GIT_COMMIT_FILE", "")
    assert configuration.model_repo() == configuration.DEFAULT_REPO
    assert configuration.model_file() == configuration.DEFAULT_FILE
