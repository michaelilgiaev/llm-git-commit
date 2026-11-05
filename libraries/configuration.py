"""configuration.py - which LOCAL GGUF model llm-git-commit runs.

Nothing is persisted: the model is a built-in default (a small CPU-friendly
Qwen2.5-Instruct GGUF) unless the environment overrides it for a run. There is no
API key any more -- inference is fully local (see llm.py).
"""

from __future__ import annotations

import os

# Default: the 1.5B Q4_K_M GGUF (~1.12 GB, ~2 GB resident) -- it writes proper
# imperative, descriptive subjects (the closest small-CPU stand-in for the old
# haiku call). On a very low-RAM box drop to the 0.5B (~491 MB, ~0.7 GB resident)
# via the env vars below -- faster and lighter, but noticeably weaker subjects:
#   LLM_GIT_COMMIT_REPO=Qwen/Qwen2.5-0.5B-Instruct-GGUF
#   LLM_GIT_COMMIT_FILE=qwen2.5-0.5b-instruct-q4_k_m.gguf
DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def model_repo() -> str:
    """Hugging Face repo id of the GGUF; $LLM_GIT_COMMIT_REPO overrides it."""
    return os.environ.get("LLM_GIT_COMMIT_REPO", "").strip() or DEFAULT_REPO


def model_file() -> str:
    """GGUF filename within the repo; $LLM_GIT_COMMIT_FILE overrides it."""
    return os.environ.get("LLM_GIT_COMMIT_FILE", "").strip() or DEFAULT_FILE
