"""MkDocs hook: render [@key, locator; @key2] citations as links to paper summaries
(or to the bibliography entry when a paper has no summary).

Reads references.bib at the repo root (the cleaned copy written by
paper-manifest). Unknown keys are logged as warnings, so `mkdocs build --strict`
fails on them. Fenced code blocks and inline code are left alone.
"""

import logging
import posixpath
import re
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.citations")

CITE = re.compile(r"\[(\s*@[\w:.+-]+[^\[\]@]*(?:@[\w:.+-]+[^\[\]@]*)*)\]")
ONE = re.compile(r"@([\w:.+-]+)\s*(?:,\s*([^;]*))?")
FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]+`")

_labels: dict = {}


def _plain(s: str) -> str:
    s = re.sub(r"\\([&%$#_{}])", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", s)
    return re.sub(r"\s+", " ", s.replace("{", "").replace("}", "")).strip()


def _field(body: str, name: str) -> str:
    m = re.search(rf"^\s*{name}\s*=\s*(\{{(?:[^{{}}]|\{{[^{{}}]*\}})*\}}|[^,\n]+)", body, re.M | re.I)
    if not m:
        return ""
    v = m.group(1).strip()
    return v[1:-1] if v.startswith("{") else v


def _label(body: str, key: str) -> str:
    authors = [a.strip() for a in re.split(r"\s+and\s+", _field(body, "author")) if a.strip()]
    names = []
    for a in authors:
        if a.startswith("{") and a.endswith("}") and "," not in a:
            names.append(_plain(a))
        else:
            names.append(_plain(a.split(",")[0] if "," in a else a.split()[-1]))
    year = _plain(_field(body, "year")) or "n.d."
    if not names:
        who = _plain(_field(body, "title")).split(":")[0][:40] or key
    elif len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} and {names[1]}"
    else:
        who = f"{names[0]} et al."
    return f"{who} {year}"


def on_config(config):
    _labels.clear()
    bib = Path(config.config_file_path).parent / "references.bib"
    if not bib.is_file():
        log.warning("references.bib not found; citations will not be linked")
        return config
    text = bib.read_text(encoding="utf-8")
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=^@|\Z)", text, re.S | re.M):
        _labels[m.group(1)] = _label(m.group(2), m.group(1))
    return config


def _items(inner: str) -> list:
    """'@a, §2; App. C; @b' -> [('a', ['§2', 'App. C']), ('b', [])]: a part without @key
    is one more locator of the key before it."""
    items = []
    for part in inner.split(";"):
        m = ONE.search(part)
        if m and part.strip().startswith("@"):
            items.append((m.group(1), [m.group(2).strip()] if (m.group(2) or "").strip() else []))
        elif items and part.strip():
            items[-1][1].append(part.strip())
    return items


def _render(match, page, files) -> str:
    here = posixpath.dirname(page.file.src_uri)
    parts = []
    for key, locs in _items(match.group(1)):
        loc = "; ".join(locs)
        if key not in _labels:
            log.warning("%s: unknown citekey @%s", page.file.src_uri, key)
            parts.append(f"**@{key}?**")
            continue
        text = _labels[key] + (f", {loc}" if loc else "")
        # Papers without a summary (e.g. no readable attachment) link to the bibliography.
        dest = f"papers/{key}.md"
        if files is not None and files.get_file_from_path(dest) is None:
            dest = f"references.md#{key}"
        target = posixpath.relpath(dest.split("#")[0], here or ".") + ("#" + key if "#" in dest else "")
        parts.append(f"[{text}]({target})")
    return "\\[" + "; ".join(parts) + "\\]" if parts else match.group(0)


def on_page_markdown(markdown, page, config, files):
    if not _labels:
        return markdown
    # Protect code, then substitute, then restore.
    saved = []

    def stash(m):
        saved.append(m.group(0))
        return f"\x00{len(saved) - 1}\x00"

    text = FENCE.sub(stash, markdown)
    text = INLINE_CODE.sub(stash, text)
    text = CITE.sub(lambda m: _render(m, page, files), text)
    return re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], text)
