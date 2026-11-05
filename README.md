# llm-git-commit

Write git commit subjects with a small language model that runs **entirely on your CPU**. No API key, no network at commit time.

Installed as a `git` shim, a bare `git commit` (with no message) generates the subject for the already-staged diff and commits it. Everything else, including `git commit -m "..."`, passes straight through to real git untouched.

```
git add -p
git commit            # the local model writes the subject, then commits
git commit -m "fix"   # unchanged -- passes through to real git
```

A progress bar (tokens written out of the budget) shows while the model writes.

## How it works

The shim inspects each `git` invocation. Only a plain `git commit` (optionally with `-a`, `-s`, `-v`, pathspecs) and **no** message source (`-m`, `-F`, `-c`, `--amend`, ...) is intercepted; the subject is generated, then handed to the real git as `-m <subject>` with your other flags, so git still does the staging and the commit. Anything else is delegated verbatim.

## Model

Inference uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (prebuilt CPU wheel, no GPU, no compiler) with a small Qwen2.5-Instruct GGUF. The file is fetched once from Hugging Face on first commit and cached; every run after is fully offline.

Default: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` / `qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1.12 GB on disk, ~2 GB resident). On a very low-RAM box, drop to the smaller, faster 0.5B model (subjects are noticeably weaker):

```
export LLM_GIT_COMMIT_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
export LLM_GIT_COMMIT_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

## Setup

Documentation only covers Arch-based systems.

Install the system packages:
```
sudo pacman --sync --refresh --needed --noconfirm - < packages_x86_64
```

Clone the repository and change into its directory:
```
git clone https://github.com/michaelilgiaev/llm-git-commit.git && \
cd llm-git-commit
```

Install (builds a local venv and writes the `git` shim + `llm-git-commit` CLI into `/usr/local/bin`):
```
bash install.sh
```

Keep this checkout in place after installing: the launchers point at its venv and `libraries/`. Ensure `/usr/local/bin` precedes `/usr/bin` on your `PATH` so the shim intercepts `git`. Then just `git commit`.

To install somewhere else:
```
PREFIX="$HOME/.local" bash install.sh    # ensure $PREFIX/bin precedes /usr/bin on PATH
```

## Tests

```
bash tests.sh
```

Pure and offline: the model is faked (no GGUF is loaded, no network), and git is exercised only against throwaway repositories under a temporary directory.
