"""llm.py - generate a commit subject with a LOCAL model, on the CPU.

No API and no network at inference time. A small Qwen2.5-Instruct GGUF runs
in-process via llama-cpp-python (CPU wheel). The file is resolved from this
repo's cache/model/, then from the Hugging Face on-disk cache, and only then
downloaded -- once, with our own byte bar, never the hub's.

Generation is one streamed pass. progress.bar shows tokens / max_tokens and
finishes on a newline. There is no second or third pass.

The llama import is deferred so importing this module stays cheap.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import configuration
import progress
from checks import RepoError, die

SYSTEM_PROMPT = (
    "Output one git commit subject. Imperative mood. At most 72 characters. "
    "No quotes, no prefix, no scope number."
)

# A subject is a short line. The cap is the bar's total, so a ramble cannot
# run the bar out and start over -- there is only this one pass.
_MAX_TOKENS = 24

# A real GGUF is about a gigabyte. Anything smaller is a partial download.
_MIN_BYTES = 1_000_000

_llama = None  # lazily built singleton (loading the GGUF costs ~1s; do it once)


def _local_file() -> Path:
    """cache/model/<file> inside this checkout. compile.sh fills it."""
    root = Path(__file__).resolve().parents[1]
    return root / "cache" / "model" / configuration.model_file()


def _cached_hf() -> str | None:
    """A GGUF already on disk from a previous hub download. No network."""
    repo = "models--" + configuration.model_repo().replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo
    if not root.is_dir():
        return None
    for path in root.glob("snapshots/*/" + configuration.model_file()):
        try:
            if path.is_file() and path.stat().st_size >= _MIN_BYTES:
                return str(path)
        except OSError:
            continue
    return None


def _download_url() -> str:
    return (
        f"https://huggingface.co/{configuration.model_repo()}"
        f"/resolve/main/{configuration.model_file()}"
    )


def _download(dest: Path) -> None:
    """Fetch the GGUF once. One byte bar, in megabytes, then a newline."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    got = 0
    try:
        req = urllib.request.Request(
            _download_url(), headers={"User-Agent": "llm-git-commit"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            progress.bar(0, max(total // (1 << 20), 1), prefix="fetching model ")
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                progress.bar(
                    got // (1 << 20),
                    max(total // (1 << 20), 1),
                    prefix="fetching model ",
                )
        if total and got < total:
            die(f"could not fetch the model ({configuration.model_file()}): "
                f"stopped at {got} of {total} bytes")
        os.replace(tmp, dest)
    except RepoError:
        raise
    except Exception as exc:
        die(f"could not fetch the model ({configuration.model_file()}): {exc}")
    finally:
        progress.done()
        if tmp.exists() and not (dest.is_file() and dest.stat().st_size >= _MIN_BYTES):
            tmp.unlink(missing_ok=True)


def _model_path() -> str:
    """Local file, else the on-disk hub cache, else one download into cache/model."""
    local = _local_file()
    if local.is_file() and local.stat().st_size >= _MIN_BYTES:
        return str(local)
    cached = _cached_hf()
    if cached:
        return cached
    _download(local)
    return str(local)


def ensure_model() -> str:
    """download_llm.sh calls this. A later `git commit` then stays offline."""
    return _model_path()


def _load():
    """Build the Llama singleton. Deferred import keeps module load cheap; verbose is
    off so the native loader banner never mixes into the progress line."""
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
    """One streamed subject for `diff`. The bar is tokens / max_tokens, drawn
    once. Returns the raw text (message.extract_subject cleans it); '' on failure.

    `model` is accepted for call-site compatibility and ignored."""
    # Load first. A download bar must finish before the token bar starts,
    # and a hub warning must not land in the middle of the line.
    try:
        llama = _load()
    except RepoError:
        raise
    except Exception:
        return ""
    text = ""
    n = 0
    progress.bar(0, max_tokens, prefix="writing commit ")
    try:
        stream = llama.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": diff},
            ],
            max_tokens=max_tokens,
            temperature=0,
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
    return text.strip()
