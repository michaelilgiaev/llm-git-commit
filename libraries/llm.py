"""llm.py - generate a commit subject with a LOCAL model, on the CPU.

No Anthropic API, no network at inference time: a small Qwen2.5-Instruct GGUF runs
in-process via llama-cpp-python (CPU wheel, no GPU). The model file is fetched once
from Hugging Face on first run (hf_hub_download) and cached; every run after is
fully offline. This is the 1:1 replacement for the old cheap-haiku subject call.

Generation is STREAMED so progress.bar can show tokens-generated / max_tokens. We
stop at the first newline, so the reply is one line -- message.extract_subject is
still the net for the rare preamble.

The heavy imports (llama_cpp, huggingface_hub) are deferred into the functions that
need them: importing this module (e.g. in the git shim's fast path, or in tests
that stub generate_subject) must stay cheap and dependency-free.
"""

from __future__ import annotations

import os

import configuration
import progress
from checks import die

SYSTEM_PROMPT = (
    "You are a git commit message generator. You receive a staged diff and output "
    "EXACTLY ONE line: the commit subject. Imperative mood, 72 characters or fewer, "
    "no trailing period, no surrounding quotes, no backticks, no emojis, no preamble, "
    "no explanation, no leading label. Output nothing except that single subject line."
)
USER_PREAMBLE = "Write the commit subject line for the staged changes below."

# A subject is a handful of tokens; cap tight so a chatty small model cannot ramble
# and the progress bar's denominator is meaningful.
_MAX_TOKENS = 48

_llama = None  # lazily built singleton (loading the GGUF costs ~1s; do it once)


def _model_path() -> str:
    """Resolve the GGUF to a local path, downloading it once if absent. $-overrides
    in configuration pick the repo/file; hf_hub_download caches under ~/.cache."""
    from huggingface_hub import hf_hub_download  # heavy; deferred

    try:
        return hf_hub_download(
            repo_id=configuration.model_repo(),
            filename=configuration.model_file(),
        )
    except Exception as exc:  # network/404/offline-first-run -> one clear message
        die(f"could not fetch the model ({configuration.model_file()}): {exc}")


def _load():
    """Build the Llama singleton. Deferred import keeps module load cheap; verbose is
    off so the native loader banner never mixes into our output."""
    global _llama
    if _llama is None:
        from llama_cpp import Llama  # heavy; deferred

        _llama = Llama(
            model_path=_model_path(),
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
    return _llama


def generate_subject(diff: str, *, model: str | None = None,
                     max_tokens: int = _MAX_TOKENS) -> str:
    """Stream a one-line commit subject for `diff` from the local model, drawing a
    progress bar (tokens / max_tokens) on stderr. Returns the RAW text
    (message.extract_subject cleans it); '' on any model failure.

    `model` is accepted for call-site compatibility but ignored -- the model is
    chosen by configuration/env, not per call."""
    text = ""
    n = 0
    progress.bar(0, max_tokens, prefix="writing commit ")
    try:
        stream = _load().create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{USER_PREAMBLE}\n\n{diff}"},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
            stop=["\n"],
            stream=True,
        )
        for chunk in stream:
            choice = chunk["choices"][0]
            piece = choice["delta"].get("content")
            if piece:
                text += piece
                n += 1
                progress.bar(n, max_tokens, prefix="writing commit ")
            if choice.get("finish_reason") is not None:
                break
    except Exception:
        return ""
    finally:
        progress.done()
    return text
