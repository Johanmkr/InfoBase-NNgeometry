# Neural Network Geometry

A textbook and literature review built from a Zotero collection of papers on the geometry of
neural networks — linear-region counting, polyhedral complexes, splines, space folding, tropical
geometry, topology, information geometry and symmetry — with every factual claim linked to the
paper it comes from.

The collection is the only input. One Better BibTeX export drives the whole site: each paper gets
a structured summary, the summaries feed a concept graph and a cross-paper comparison, and the
concept graph drives the chapters. Pages are written by AI agents from the papers themselves, and
the Markdown is checked for citation consistency, math that will not render and stale content.

**The cited papers are the authority.** Treat the site as a study aid, not as a substitute for
reading them.

## Read it

```bash
uv run mkdocs serve        # http://127.0.0.1:8000
```

Without `uv`: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/mkdocs serve`.
Commit `uv.lock`; the Pages workflow builds with `uv sync --locked`.

## What is in it

| Path | Contents |
|---|---|
| `docs/textbook/` | 14 chapters in teaching order, each with exercises |
| `docs/papers/` | 41 structured summaries, one per paper |
| `docs/comparison/` | Concept map, timeline, comparison tables and themed essays |
| `docs/notation.md` | The one notation used across the textbook, mapped to each paper's own |
| `docs/references.md` | Bibliography, generated from the collection |
| `state/concepts.yml` | Concept graph: each concept's prerequisites and which papers introduce or use it |
| `state/outline.yml` | The chapter plan the textbook is generated from |
| `state/proofread.md` | The last proofreading report, including findings not yet applied |

Chapter order is the reading order: polyhedra and arrangements → the region complex → counting
regions → space folding → splines → tropical geometry → region topology → enumeration and
verification → information geometry → manifold learning → latent geometry → metric flows →
symmetry and geometric deep learning → implicit and geometry-informed representations.

## How it is built

`literature.yml` names the collection and its raw export. The pipeline is the `textbook-sync`
skill, which runs on a `sync/<date>` branch and commits there for review; it never pushes.

```
Zotero collection  →  NN-geometry.bib (raw, git-ignored)
                            │
                   summaries in docs/papers/
                            │
            concept graph (state/concepts.yml) ─ notation
                            │
                 outline (state/outline.yml)
                            │
                  chapters + exercises, comparison pages
                            │
                audit → proofread → mkdocs build --strict
```

- **After changing the collection**, run `/textbook-sync`. It detects added, changed and removed
  papers, re-summarises only those, updates everything downstream, and commits to a `sync/` branch.
- **Asking questions about the material**: use the `literature-teacher` agent, which answers from
  the collection and cites it.

## Conventions to respect

- **Never edit `NN-geometry.bib`** — Zotero rewrites it — and never commit it: its `file` fields
  hold local paths. `references.bib` is the cleaned, committed copy.
- Cite with `[@citekey]` or `[@citekey, §3.2]`; `hooks/citations.py` turns those into links to the
  paper's summary. Do not write the link by hand.
- Linked pages are generated. Files under `docs/` and `state/` carry `built_from` / `source_sha256`
  front matter, which the sync uses to find stale pages — do not remove it. Several are also
  generated outright (`docs/textbook/index.md`, `docs/papers/index.md`, `docs/references.md`, the
  `*-nav` blocks of `mkdocs.yml`), so edit their source rather than the file.

See `AGENTS.md` for the rules agents follow in this repo.

## Publishing

GitHub Pages builds from `main` (`.github/workflows/docs.yml`) with `mkdocs build --strict`, so a
broken link or an undefined macro fails the build. To publish a reviewed sync branch:

```bash
git switch main && git merge --ff-only sync/<date> && git push
```

## Requirements

- Three packages: `mkdocs`, `mkdocs-material`, `pyyaml` (see `pyproject.toml`; `requirements.txt`
  is the pip fallback and the two are kept in sync).
- The scripts under the literature skills declare their own dependencies (PEP 723), so `uv run`
  provides them without installing anything.
- `uv.lock` is committed and the Pages build uses it (`uv sync --locked`); keep it in sync when
  dependencies change.
