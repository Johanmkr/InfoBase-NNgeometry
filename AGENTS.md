# Neural Network Geometry: literature textbook

This repo turns the Zotero collection exported to `NN-geometry.bib` into paper
summaries, comparisons and a textbook, published with MkDocs Material on
GitHub Pages. It follows the `academic-md-conventions` skill; read it before
changing anything under `docs/` or `state/`.

- **Update after changing the Zotero collection:** run the `textbook-sync` skill
  (`/textbook-sync`). It works on a `sync/<date>` branch and commits there; the
  owner reviews, merges and pushes.
- **Ask questions about the material:** use the `literature-teacher` agent.
- Never edit `NN-geometry.bib` (Zotero rewrites it) and never commit it (it holds local
  file paths). `references.bib` is the cleaned, committed copy.
- Generated files carry `built_from` / `source_sha256` front matter. Do not
  remove it; the sync uses it to find stale pages.
- Read the site locally: `uv run mkdocs serve`, then open http://127.0.0.1:8000
  (without uv: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/mkdocs serve`).
  Commit `uv.lock`; the Pages workflow builds with `uv sync --locked`.
