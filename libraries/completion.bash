# Bash completion for the `llm-git-commit` CLI.
#
# Installed where the bash-completion lazy loader finds it (see README):
#   /usr/share/bash-completion/completions/llm-git-commit
# The git shim itself rides git's OWN completion, so this only covers the CLI.
# Position 1 completes the subcommand list (commit, help); nothing else completes.

_llm_git_commit_complete() {
    local cur
    cur="${COMP_WORDS[COMP_CWORD]}"
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "commit help" -- "$cur") )
    fi
    return 0
}

complete -F _llm_git_commit_complete llm-git-commit
