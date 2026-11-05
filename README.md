<p align="center">
  <img src="assets/llm-git-commit_title_627x230.png" alt="llm-git-commit">
</p>

Installed as a `git` shim, a bare `git commit` with no message writes the subject for the already staged diff with a small language model that runs entirely on your CPU, then commits. Everything else, including `git commit -m "..."`, passes straight through to real git. No API key, no network at commit time. A progress bar shows while the model writes. Keep this checkout in place; the installed commands run from it. Remove everything with `llm-git-commit --uninstall`.

## Setup

Documentation only covers Arch-based systems.

```
sudo pacman --sync --refresh --needed --noconfirm - < packages_x86_64 && \
bash download_llm.sh && \
bash compile.sh && \
sudo install -m 755 output/git /usr/local/bin/ && \
sudo install -m 755 output/llm-git-commit /usr/bin/ && \
sudo install -m 644 libraries/completion.bash /usr/share/bash-completion/completions/llm-git-commit
```
