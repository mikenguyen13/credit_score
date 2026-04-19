# Credit Scoring: Theory, Methods, and Practice

**Read the book online:** https://mikenguyen13.github.io/credit_score/

A working, executable, publishable textbook on credit scoring. Covers classical statistics, machine learning, deep learning, alternative data, explainability, fairness, causal inference, graph neural networks, LLMs, regulatory capital, and production deployment. Every example runs on publicly available data.

Book source lives in [`book/`](book/). The book is built with [Quarto](https://quarto.org).

## Quickstart

```bash
# 1. Install uv and Quarto (see Appendix B of the book for alternatives).
# 2. Clone this repository, then:

uv sync --python 3.12
uv run python -m ipykernel install --user --name credit-scoring-book --display-name "Python (credit-scoring-book)"

# 3. Render the book:
cd book
quarto render
```

Rendered HTML lands in `book/_book/`. PDF and EPUB are produced from the same sources via `quarto render --to pdf` and `--to epub`.

## Repository layout

```
credit_score/
├── pyproject.toml          # Python environment (uv-managed)
├── LICENSE                 # CC-BY-4.0 for text, MIT for code
├── README.md
└── book/
    ├── _quarto.yml         # Book configuration
    ├── index.qmd           # Preface
    ├── references.bib      # Top-tier citations
    ├── references.qmd      # Rendered reference list
    ├── chapters/           # 35 chapters
    ├── appendices/         # Math, env setup, data catalog
    ├── code/               # Shared Python helpers (csutils.py)
    ├── data/               # Downloaded datasets (gitignored)
    ├── deployment/         # FastAPI, Docker, MLflow recipes
    ├── figures/
    └── styles.css
```

## macOS note

Gradient boosting libraries require `libomp`. On macOS, install via Homebrew (`brew install libomp`) or use the R project's pre-built binary. The environment setup appendix has the full recipe.

## Publishing to GitHub Pages

The repository ships source only. Rendered HTML, PDF, EPUB, execution caches, and downloaded datasets are gitignored so the clone stays light.

To publish the book to GitHub Pages:

```bash
cd book
quarto publish gh-pages
```

This renders the book locally, pushes the output to an orphan `gh-pages` branch, and configures Pages automatically on first run. In the GitHub repo settings, confirm **Pages → Source = `gh-pages` branch, `/ (root)`**.

Re-run `quarto publish gh-pages` after any content change to refresh the site.

## License

Text: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Code: MIT. See [LICENSE](LICENSE).
