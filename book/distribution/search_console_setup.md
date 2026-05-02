# One-time search console + IndexNow setup

These three actions take ~5 minutes total, are free, and account for ~80% of the discoverability gain. They cannot be fully automated because they require *your* account auth — but each is a single click after the first.

---

## 1. Google Search Console (3 minutes)

1. Open https://search.google.com/search-console
2. Click **Add property** → **URL prefix** → enter `https://mikenguyen13.github.io/credit_score/` → Continue.
3. Verify ownership. Easiest method: **HTML tag**. Google will show a tag like `<meta name="google-site-verification" content="ABC123...">`.
4. Tell Claude the verification code, or paste it into [book/_includes/json-ld.html](../_includes/json-ld.html) at the top.
5. Re-render and publish (`quarto publish gh-pages`).
6. Back in Search Console, click **Verify**.
7. Once verified, go to **Sitemaps** → enter `sitemap.xml` → Submit.

After that, every render of the book auto-pings Google because Google polls verified sitemaps. Zero ongoing work.

## 2. Bing Webmaster Tools (90 seconds)

1. Open https://www.bing.com/webmasters
2. Sign in with any Microsoft / Google / Facebook account.
3. **Add a site** → enter `https://mikenguyen13.github.io/credit_score/`. Easiest: **Import from Google Search Console** if you did step 1 — auto-imports verification + sitemap.
4. Otherwise, verify with the same `<meta>` tag method as above (Bing shows you a `msvalidate.01` tag).
5. After verification: **Sitemaps** → submit `sitemap.xml`.

Bing also feeds DuckDuckGo, Yahoo, Ecosia, and (importantly) ChatGPT search and Copilot.

## 3. IndexNow (already wired up — just generate a key)

```bash
cd book
python -c "import secrets; print(secrets.token_hex(16))" > .indexnow.key
```

Add to `.gitignore` so the key is not public **only if you treat it as a secret**. The IndexNow protocol actually requires the key to be publicly fetchable at `https://mikenguyen13.github.io/credit_score/<key>.txt` — so the key is not a credential, it just needs to match. The script auto-creates that verification file. You can safely commit `.indexnow.key` if you want.

After every publish:

```bash
cd book
quarto publish gh-pages
python scripts/indexnow_ping.py
```

This single ping notifies Bing, Yandex, Seznam, Naver, Yep.com, and any other IndexNow-participating engine within minutes. Free, no account.

---

## Optional: Yandex Webmaster, Naver Search Advisor, Baidu Webmaster

If you want explicit coverage in Russia / Korea / China search rankings (vs relying on the wildcard crawl):

- Yandex: https://webmaster.yandex.com — submit sitemap. Honors IndexNow already, so the script in step 3 covers re-crawl signals.
- Naver: https://searchadvisor.naver.com — submit sitemap, verify by meta tag.
- Baidu: https://ziyuan.baidu.com — submit sitemap; requires a Chinese phone number for full features.

For most authors writing in English, Google + Bing + IndexNow capture > 95% of organic and AI traffic. Skip the rest unless you have a specific reason.
