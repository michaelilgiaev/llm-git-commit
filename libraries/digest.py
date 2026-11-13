"""digest.py - the commit line, and the check on a model's rephrase.

The line itself is written by harness.py. This module keeps `line` and `choose`
so callers and tests have one import. `choose` still guards a rephrase of an
older digest line. A fact card is checked by harness.pick.
"""

from __future__ import annotations

import re
from pathlib import Path

import harness

# Syntax and test-harness words. These are not "what changed".
_STOP = {
    "and", "any", "are", "async", "await", "bool", "break", "byte", "bytes",
    "case", "chan", "class", "clone", "const", "continue", "def", "default",
    "defer", "elif", "else", "enum", "err", "error", "expect", "false", "fn",
    "for", "from", "func", "function", "got", "impl", "import", "inner", "int",
    "int32", "int64", "interface", "into", "len", "let", "make", "map", "mut",
    "new", "nil", "none", "not", "null", "ok", "opts", "package", "path", "pub",
    "public", "range", "return", "select", "self", "some", "static", "string",
    "struct", "switch", "the", "this", "true", "tt", "type", "uint", "unwrap",
    "use", "val", "var", "void", "want", "while", "with",
    "add", "drop", "update", "then", "in",
}
_GENERIC_FUNCS = {
    "init", "update", "string", "main", "test", "setup", "teardown",
    "get", "set", "run",
}

_FUNC = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?"
    r"(?:async\s+)?"
    r"(?:def|fn|func|class|struct|enum)\s+"
    r"(?:\([^)]*\)\s*)?"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_OURS = re.compile(r"\bin [\w./+-]+\.\w+")
_STAT_FILE = re.compile(r"^\s*(.+?)\s+\|")

# A parsed patch bigger than this is cut. The line only needs the first
# files; the model never sees this text.
_PARSE_CAP = 40_000
_MAX_FILES = 8
_MAX_LINE = 360


class _File:
    def __init__(self) -> None:
        self.path = ""
        self.added: list[str] = []
        self.removed: list[str] = []
        self.hunk = ""


def line(stat: str, patch: str) -> str:
    """The harness's commit line. Qwen may rephrase it; this is the fallback."""
    return harness.subject(stat, patch)


def choose(model_line: str, source: str) -> str:
    """Keep a rewrite that still names the files and the one primary symbol.

    Helper names may be dropped. Otherwise keep `source`. A source that is
    not one of our lines is returned unchanged only through `model_line`.
    """
    if not _OURS.search(source or ""):
        return model_line
    if model_line and _faithful(model_line, source):
        return model_line
    return source


def _faithful(model_line: str, source: str) -> bool:
    """True when the rewrite keeps the basenames and the primary symbol.

    Several behavior clauses must stay several clauses. Two or more new
    code-shaped names (snake_case or an internal capital, length at least 6)
    mean the rewrite left the diff. Ordinary words do not count: a high-level
    line is allowed to say why.
    """
    folded = model_line.casefold()
    for tok in _required(source):
        if tok.casefold() not in folded:
            return False
    if "; " in source and "; " not in model_line:
        return False
    extras = [
        tok for tok in _idents(model_line)
        if tok.casefold() not in source.casefold()
        and len(tok) >= 6
        and _symbolish(tok)
    ]
    return len(extras) < 2


def _symbolish(tok: str) -> bool:
    return "_" in tok or any(ch.isupper() for ch in tok[1:])


def _required(source: str) -> list[str]:
    """File basenames, plus the first real symbol. Later helpers are optional."""
    names = re.findall(r"[\w./+-]+\.\w+", source)
    stems = {Path(name).stem.casefold() for name in names}
    stems |= {Path(name).name.casefold() for name in names}
    plain: list[str] = []
    primary = ""
    for tok in _idents(source):
        if tok.casefold() in _STOP or tok.casefold() in stems:
            continue
        if _symbolish(tok):
            primary = tok
            break
        plain.append(tok)
    if not primary and plain:
        primary = plain[0]
    seen: list[str] = []
    for tok in names + ([primary] if primary else []):
        if tok and tok not in seen:
            seen.append(tok)
    return seen


def _idents(text: str) -> list[str]:
    return _IDENT.findall(text)


def _parse(patch: str) -> list[_File]:
    files: list[_File] = []
    cur: _File | None = None
    for raw in (patch or "")[:_PARSE_CAP].splitlines():
        if raw.startswith("diff --git "):
            cur = _File()
            files.append(cur)
            continue
        if cur is None:
            continue
        if raw.startswith("+++ "):
            path = raw[4:]
            if path.startswith("b/"):
                path = path[2:]
            if path != "/dev/null":
                cur.path = path
            continue
        if raw.startswith("@@"):
            cur.hunk = raw
            continue
        if raw.startswith("+"):
            cur.added.append(raw[1:])
        elif raw.startswith("-") and not raw.startswith("---"):
            cur.removed.append(raw[1:])
    return [item for item in files if item.path]


def _main_file(files: list[_File]) -> _File:
    for item in files:
        if not _is_test(item.path):
            return item
    return files[0]


def _second_file(files: list[_File], primary: _File) -> _File | None:
    """One follow-up, preferring the test file over another helper."""
    rest = [item for item in files if item is not primary]
    for item in rest:
        if _is_test(item.path):
            return item
    return rest[0] if rest else None


def _is_test(path: str) -> bool:
    name = Path(path).name.casefold()
    if name.startswith("test_") or name.startswith("test."):
        return True
    return name.endswith(("_test.go", "_test.rs", "_test.py", "_test.ts", "_test.js"))


def _clause(item: _File, *, follow: bool = False) -> str:
    """One verb and one symbol. A follow-up file does not list helpers."""
    added_funcs = _prefer_strong(_funcs(item.added))
    removed_funcs = [
        name for name in _prefer_strong(_funcs(item.removed)) if name not in added_funcs
    ]
    added = _tokens(item.added)
    removed = set(_tokens(item.removed))
    fresh = [tok for tok in added if tok not in removed and tok not in added_funcs]
    name = Path(item.path).name
    symbol = added_funcs[0] if added_funcs else (fresh[0] if fresh else "")
    note = _best_comment(item.added, symbol)
    if note:
        if symbol and symbol.casefold() not in note.casefold() and not follow:
            what = f"{note.rstrip('.')} ({symbol})"
        else:
            what = note.rstrip(".")
        return f"{what} in {name}"
    if follow:
        if added_funcs and not _looks_like_test(added_funcs[0]):
            return f"add {added_funcs[0]} in {name}"
        if added_funcs and _looks_like_test(added_funcs[0]):
            return f"cover this in {name}"
        if removed_funcs:
            return f"drop {removed_funcs[0]} in {name}"
        return f"update {name}"
    if added_funcs:
        what = "add " + added_funcs[0]
    elif removed_funcs and not fresh:
        what = "drop " + removed_funcs[0]
    elif fresh:
        what = ("fix " if item.removed else "add ") + fresh[0]
    elif removed_funcs:
        what = "drop " + removed_funcs[0]
    else:
        hunk = _hunk_name(item.hunk)
        what = f"fix {hunk}" if hunk else ""
    if not what:
        return f"update {name}"
    return f"{what} in {name}"


def _best_comment(lines: list[str], symbol: str = "") -> str:
    """The added sentence that best says what changed.

    Build tags and one-word notes are skipped. Prefer a sentence that names
    the symbol and states a behavior, then cut it to one clause.
    """
    chunks: list[str] = []
    buf: list[str] = []
    for raw in lines:
        match = re.match(r"^(?://+|#+|/\*+|\*)\s?(.*)", raw.strip())
        if not match:
            if buf:
                chunks.append(" ".join(buf))
                buf = []
            continue
        piece = match.group(1).strip().strip("*/").strip()
        if not piece or piece.startswith(("go:build", "nolint", "+build")):
            continue
        buf.append(piece)
    if buf:
        chunks.append(" ".join(buf))
    chunks.extend(_doc_chunks(lines))
    best = ""
    best_score = 0
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!])\s+", chunk):
            sentence = " ".join(sentence.replace("``", "").replace("`", "").split())
            score = _comment_score(sentence, symbol)
            if score > best_score:
                best, best_score = sentence, score
    if not best:
        return ""
    # A huge test name in front of "is a regression test for ..." hides the
    # behavior. Keep the behavior.
    named = re.match(r"^[A-Za-z_][A-Za-z0-9_]{24,}\s+is\s+(.*)", best)
    if named:
        best = named.group(1)
    if len(best) > 110:
        best = best[:110].rsplit(" ", 1)[0]
    if best.endswith("."):
        best = best[:-1]
    first, sep, rest = best.partition(" ")
    if first.isupper() or (len(first) > 1 and first[0].isupper() and first[1:].islower()):
        best = first.lower() + sep + rest
    return best


def _looks_like_test(name: str) -> bool:
    return name.startswith(("test_", "Test", "test"))


def _doc_chunks(lines: list[str]) -> list[str]:
    """Sentences from added Python docstrings. Comments are handled separately."""
    chunks: list[str] = []
    buf: list[str] = []
    fence = ""

    def finish() -> None:
        cleaned = _clean_doc(" ".join(buf))
        if cleaned:
            chunks.append(cleaned)

    for raw in lines:
        text = raw.strip()
        if text.startswith((":param", ":return", ":rtype", ":raises", ":type")):
            continue
        if not fence:
            match = re.match(r"^[rubfRUBF]*(\"\"\"|''')(.*)$", text)
            if not match:
                continue
            fence = match.group(1)
            rest = match.group(2)
            if fence in rest:
                body = rest[: rest.find(fence)]
                cleaned = _clean_doc(body)
                if cleaned:
                    chunks.append(cleaned)
                fence = ""
                continue
            if rest:
                buf.append(rest)
            continue
        if fence in text:
            before = text[: text.find(fence)]
            if before:
                buf.append(before)
            finish()
            buf = []
            fence = ""
            continue
        if text:
            buf.append(text)
    return chunks


def _clean_doc(text: str) -> str:
    text = text.replace("``", "").replace("`", "")
    text = re.sub(r":(?:meth|class|func|attr|exc|paramref|mod):", "", text)
    return " ".join(text.split())


def _comment_score(sentence: str, symbol: str) -> int:
    words = sentence.split()
    if len(words) < 6 or "://" in sentence[:16]:
        return 0
    score = min(len(words), 16)
    folded = sentence.casefold()
    if symbol and symbol.casefold() in folded:
        score += 4
    if folded.startswith("without "):
        score -= 3
    if "human-readable" in folded or "unix implementation" in folded or folded.startswith("on windows"):
        score -= 5
    for cue in ("instead", "when", "avoid", "fallback", "always", "never", "outside"):
        if cue in folded:
            score += 2
    return score


def _prefer_strong(names: list[str]) -> list[str]:
    """Drop short generic declarations when a real symbol is also there.

    `def hi` stays (it is the only name). `def cli` next to a long test name
    does not.
    """
    strong = [name for name in names if _strong(name)]
    return strong or names


def _funcs(lines: list[str]) -> list[str]:
    found: list[str] = []
    for raw in lines:
        text = raw.strip()
        match = _FUNC.match(text)
        if not match:
            continue
        name = match.group(1)
        if name not in found and _keep_func(name):
            found.append(name)
    return found


def _keep_func(name: str) -> bool:
    """Declared names are the change, even when short. Generic methods are not."""
    folded = name.casefold()
    if len(name) < 2 or folded in _STOP or folded in _GENERIC_FUNCS:
        return False
    return True


def _strong(tok: str) -> bool:
    """A symbol worth naming: snake_case, or an internal capital.

    Plain words ('partition', 'size', 'Init') are how code is written, not
    what the commit did.
    """
    if len(tok) < 4 or tok.casefold() in _STOP:
        return False
    if "_" in tok:
        return True
    return any(ch.isupper() for ch in tok[1:])


def _hunk_name(header: str) -> str:
    match = re.search(r"@@ .*@@\s*(.*)", header)
    if not match:
        return ""
    names = [tok for tok in _IDENT.findall(match.group(1)) if _strong(tok)]
    return names[-1] if names else ""


def _tokens(lines: list[str]) -> list[str]:
    seen: list[str] = []
    bag: set[str] = set()
    for raw in lines:
        if not _interesting(raw):
            continue
        for tok in _IDENT.findall(raw):
            if tok in bag or not _strong(tok):
                continue
            bag.add(tok)
            seen.append(tok)

    def rank(pair: tuple[int, str]) -> tuple[int, int]:
        index, tok = pair
        score = 0
        if "_" in tok or any(ch.isupper() for ch in tok[1:]):
            score += 2
        if len(tok) >= 8:
            score += 1
        return (-score, index)

    return [tok for _, tok in sorted(enumerate(seen), key=rank)]


def _interesting(raw: str) -> bool:
    text = raw.strip()
    if len(text) < 3:
        return False
    if text[0] in "#/*" or text.startswith(("//", "import ", "from ", "package ", "use ")):
        return False
    return any(ch.isalnum() for ch in text)


def _from_stat(stat: str) -> list[str]:
    names: list[str] = []
    for raw in (stat or "").splitlines():
        match = _STAT_FILE.match(raw)
        if not match:
            continue
        path = match.group(1).strip().strip('"')
        if " => " in path:
            path = path.split(" => ")[-1].strip().strip("}")
        name = Path(path).name
        if name and name not in names:
            names.append(name)
    return [f"update {name}" for name in names[:_MAX_FILES]]
