"""Generate llms.txt and llms-full.txt from _quarto.yml + chapter sources.

Both files regenerate on every Quarto build via the project pre-render hook,
so adding/removing/renaming chapters in _quarto.yml automatically updates the
LLM-discovery index and the full-content bundle.

- llms.txt:       llmstxt.org spec, concise index of all parts/chapters.
- llms-full.txt:  full prose of every chapter and appendix, code chunks
                  stripped, optimized for LLM ingestion.

Run from book/ directory or as a Quarto pre-render hook:
    python scripts/build_llms_full.py
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SITE_URL = "https://mikenguyen13.github.io/credit_score"
QUARTO_YML = ROOT / "_quarto.yml"

YAML_FRONT = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
HEADING_H1 = re.compile(r"^#\s+([^\n{]+?)(\s*\{[^}]*\})?\s*$", re.MULTILINE)


def strip_code_blocks(text: str) -> str:
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            continue
        out.append(line)
    return "\n".join(out)


def clean_prose(text: str) -> str:
    text = YAML_FRONT.sub("", text, count=1)
    text = strip_code_blocks(text)
    text = re.sub(r"\{#[^}]+\}", "", text)
    text = re.sub(r"::: \{[^}]*\}\n", "", text)
    text = re.sub(r":::\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def first_heading(qmd_path: Path) -> str:
    text = qmd_path.read_text(encoding="utf-8")
    text = YAML_FRONT.sub("", text, count=1)
    m = HEADING_H1.search(text)
    if m:
        return m.group(1).strip()
    return qmd_path.stem.replace("-", " ").title()


def html_url(qmd_rel: str) -> str:
    return f"{SITE_URL}/{qmd_rel.replace('.qmd', '.html')}"


def parse_book_structure() -> tuple[dict, list[tuple[str, list[str]]], list[str]]:
    """Return (book_meta, parts=[(part_title, [qmd_rel,...]), ...], appendices=[qmd_rel,...])."""
    cfg = yaml.safe_load(QUARTO_YML.read_text(encoding="utf-8"))
    book = cfg.get("book", {})
    chapters_node = book.get("chapters", [])
    parts: list[tuple[str, list[str]]] = []
    flat_top: list[str] = []
    for entry in chapters_node:
        if isinstance(entry, str):
            flat_top.append(entry)
        elif isinstance(entry, dict):
            title = entry.get("part", "")
            sub = [c for c in entry.get("chapters", []) if isinstance(c, str)]
            parts.append((title, sub))
    if flat_top:
        parts.insert(0, ("", flat_top))
    appendices = [a for a in book.get("appendices", []) if isinstance(a, str)]
    return book, parts, appendices


def build_llms_index(book: dict, parts, appendices) -> str:
    title = book.get("title", "Book")
    description = (book.get("description") or "").strip()
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    if description:
        for para in description.split("\n\n"):
            lines.append(f"> {para.strip()}")
            lines.append(">")
        lines.pop()
    lines.append("")
    lines.append(
        "Author: Mike Nguyen. License: CC-BY-4.0 for text, MIT for code. "
        "Citation, ingestion, and adaptation by humans and AI systems are "
        "explicitly welcome with attribution."
    )
    lines.append("")
    lines.append(
        "The book is a Quarto project. Each chapter is an executable `.qmd` "
        "file with runnable Python on publicly available data."
    )
    lines.append("")

    for part_title, qmd_list in parts:
        if part_title:
            lines.append(f"## {part_title}")
        else:
            lines.append("## Front matter")
        lines.append("")
        for qmd in qmd_list:
            qmd_path = ROOT / qmd
            if not qmd_path.exists():
                continue
            heading = first_heading(qmd_path)
            url = html_url(qmd)
            lines.append(f"- [{heading}]({url})")
        lines.append("")

    if appendices:
        lines.append("## Appendices")
        lines.append("")
        for qmd in appendices:
            qmd_path = ROOT / qmd
            if not qmd_path.exists():
                continue
            heading = first_heading(qmd_path)
            url = html_url(qmd)
            lines.append(f"- [{heading}]({url})")
        lines.append("")

    lines.append("## Optional")
    lines.append("")
    lines.append(f"- [Source repository]({book.get('repo-url', '')})")
    lines.append(f"- [References]({SITE_URL}/references.html)")
    lines.append(f"- [Full content for LLM ingestion]({SITE_URL}/llms-full.txt)")
    lines.append(f"- [Sitemap]({SITE_URL}/sitemap.xml)")
    lines.append("")
    return "\n".join(lines)


def build_llms_full(book: dict, parts, appendices) -> str:
    parts_out: list[str] = []
    parts_out.append(f"# {book.get('title', 'Book')}")
    parts_out.append("")
    parts_out.append(f"Source: {SITE_URL}")
    parts_out.append("Author: Mike Nguyen")
    parts_out.append("License: CC-BY-4.0 (text), MIT (code)")
    parts_out.append("")
    parts_out.append(
        "This file is a single-document ingestion bundle for LLMs. It contains "
        "the full prose of every chapter and appendix with executable code "
        "chunks stripped. For runnable code, see the GitHub repository: "
        f"{book.get('repo-url', '')}"
    )
    parts_out.append("")
    parts_out.append("=" * 80)

    ordered: list[str] = []
    for _title, qmds in parts:
        ordered.extend(qmds)
    ordered.extend(appendices)

    for rel in ordered:
        path = ROOT / rel
        if not path.exists():
            continue
        text = clean_prose(path.read_text(encoding="utf-8"))
        parts_out.append("")
        parts_out.append("=" * 80)
        parts_out.append(f"# Source: {rel}")
        parts_out.append("=" * 80)
        parts_out.append("")
        parts_out.append(text)
        parts_out.append("")
    return "\n".join(parts_out)


def main() -> None:
    book, parts, appendices = parse_book_structure()

    llms_idx = build_llms_index(book, parts, appendices)
    (ROOT / "llms.txt").write_text(llms_idx, encoding="utf-8")

    llms_full = build_llms_full(book, parts, appendices)
    (ROOT / "llms-full.txt").write_text(llms_full, encoding="utf-8")

    idx_kb = (ROOT / "llms.txt").stat().st_size / 1024
    full_kb = (ROOT / "llms-full.txt").stat().st_size / 1024
    print(f"wrote llms.txt ({idx_kb:.1f} KB), llms-full.txt ({full_kb:.1f} KB)")


if __name__ == "__main__":
    main()
