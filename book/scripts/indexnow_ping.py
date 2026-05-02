"""Ping IndexNow API after a publish so Bing, Yandex, Seznam, Naver, and other
participating engines re-crawl quickly. Free, no account, no payment.

One-time setup
--------------
1. Generate an API key (any 8-128 hex chars). The script reads it from the
   environment variable INDEXNOW_KEY, or from a file `book/.indexnow.key`.
   Easy way to generate one:
       python -c "import secrets; print(secrets.token_hex(16))"
2. Place a verification file at the site root containing only the key:
       book/<key>.txt   with content == <key>
   (This file ships with the rest of `_book/` because it's listed in
   `resources` of _quarto.yml; see the auto-create step below.)
3. Run after each publish:
       python scripts/indexnow_ping.py

The script will:
- Read the key.
- Auto-create the verification file at book/<key>.txt if missing.
- POST the full URL list (parsed from sitemap.xml in _book/) to the IndexNow
  endpoint, which forwards to all participating search engines.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_HOST = "mikenguyen13.github.io"
SITE_URL = f"https://{SITE_HOST}/credit_score"
SITEMAP = ROOT / "_book" / "sitemap.xml"
KEY_FILE = ROOT / ".indexnow.key"
ENDPOINT = "https://api.indexnow.org/IndexNow"


def load_key() -> str:
    key = os.environ.get("INDEXNOW_KEY")
    if key:
        return key.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    sys.exit(
        "No IndexNow key found. Generate one with:\n"
        "  python -c \"import secrets; print(secrets.token_hex(16))\" > book/.indexnow.key"
    )


def ensure_verification_file(key: str) -> None:
    target = ROOT / f"{key}.txt"
    if not target.exists():
        target.write_text(key, encoding="utf-8")
        print(f"created verification file: {target.relative_to(ROOT)}")


def parse_sitemap() -> list[str]:
    if not SITEMAP.exists():
        sys.exit(f"sitemap.xml not found at {SITEMAP}. Run `quarto render` first.")
    tree = ET.parse(SITEMAP)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text.strip() for loc in tree.findall(".//s:url/s:loc", ns) if loc.text]


def submit(key: str, urls: list[str]) -> None:
    payload = {
        "host": SITE_HOST,
        "key": key,
        "keyLocation": f"{SITE_URL}/{key}.txt",
        "urlList": urls,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"IndexNow status: {resp.status} {resp.reason} ({len(urls)} URLs)")


def main() -> None:
    key = load_key()
    ensure_verification_file(key)
    urls = parse_sitemap()
    print(f"submitting {len(urls)} URLs to IndexNow...")
    submit(key, urls)


if __name__ == "__main__":
    main()
