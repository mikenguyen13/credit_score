# Fabrication Audit: references.bib + refs/ch-*.bib

Date: 2026-04-23 (original) / 2026-04-30 (cleanup pass)
Method: 9 parallel AI-verification agents ran DOI / arXiv / WebFetch / WebSearch
lookups across ~1,320 entries in `book/references.bib` and ~981 entries across the
38 per-chapter / appendix bib files under `book/refs/`.

Entries classified as **LIKELY-FABRICATED** or **UNCERTAIN**. Verified entries
are not listed. Canonical classics (Fisher, Altman, Beaver, Merton, Cox 1972,
Breiman, Hastie/Tibshirani/Friedman, Hand & Henley, Thomas, Friedman GBM,
LeCun/Bengio/Hinton, Goodfellow, Rumelhart, Hochreiter, Pedregosa, Chen & Guestrin)
were marked verified without lookup.

## Status as of 2026-04-30

- **All 32 LIKELY-FABRICATED entries**: deleted from `references.bib` and chapter bibs;
  prose citations removed or replaced. Confirmed via grep across active files.
- **42 UNCERTAIN entries**: 28 metadata-fixed in earlier passes; 14 misleading bibkeys
  renamed to match real publication years/topics on 2026-04-30 (see "Renames" below).
- **9 additional cleanup deletions on 2026-04-30**: `assef2024realtime`,
  `chu2024vietnam`, `armstrong2013flow`, `breeden2024xai`, `breeden2012vintage`,
  `kuck2003identifying`, `alonso2020emfintech`, `crosbie2003modeling`,
  `korablev2009validation`. Plus duplicates `chen2019howvaluable`,
  `jagtiani2019roles_alt`, `veldkamp2023valuing` removed.
- **2 working-paper → published-version updates**: `bhutta2022how` now points to
  Journal of Finance 2025; `babina2024customer` now points to JFE 2025.
- **1 entry replaced**: `agarwal2020bandit` (mislabeled CACM 2022 paper) replaced with
  two real Agarwal et al. NeurIPS bandit papers (`agarwal2020choicebandits`,
  `agarwal2024network`).
- **1 URL fix**: `gabaix2024llm` URL corrected from NBER w32768 to w32001.
- **74 new top-journal entries** added across 13 chapter bibs and woven into chapter
  prose ("Further reading" narrative paragraphs) on 2026-04-30. See changelog in
  this commit's diff for the full enumeration.

### Renames applied 2026-04-30 (bibkey now matches real year/title)

| Old bibkey | New bibkey | Real venue |
|------------|------------|------------|
| `breeden2019survey` | `breeden2020survey` | J. Credit Risk 2020 |
| `gelman2017prior` | `gelman2008prior` | AoAS 2008 |
| `foster1989resolving` | `foster2004variable` | JASA 2004 |
| `chen2023nonbank` | `chen2019fintech` | RFS 2019 |
| `rona2021big` | `rona2020predicting` | SER 2020 |
| `miller2017impact` | `miller2018privacy` | MS 2018 |
| `goldfarb2023regulation` | `goldfarb2011privacy` | MS 2011 |
| `lowry2020initial` | `lowry2017ipo` | FnT 2017 |
| `firth2021readability` | `firth2015corporate` | MS 2015 |
| `liu2021ptuning` | `liu2024gpt` | AI Open 2024 |
| `kraus2013fairness` | `kraus2017decision` | DSS 2017 |
| `navas2016optimal` | `navas2020optimal` | arXiv 2020 |
| `bauguess2020role` | `bauguess2017role` | SEC 2017 |
| `hamilton2024graphnn` | `liu2019geniepath` | AAAI 2019 |
| `hue2023xper` | (kept, year 2025 → 2022) | arXiv 2022 |

### Final bib counts as of 2026-04-30

- `references.bib`: 1284 entries
- per-chapter bibs total: 1400 entries
- Orphan citation check: 0 unresolved `@key` references in chapter prose
  (excluding crossref prefixes `fig-`, `tbl-`, `sec-`, `eq-`, `lst-`, etc.)

The original lists below are preserved as historical record. They no longer reflect
the live bib state; cross-reference against the cleanup status above.

Per-agent reports:
- references.bib: `citation_check_part{1..6}.md`
- per-chapter bibs: `citation_check_chapters_{a,b,c}.md`

---

## LIKELY-FABRICATED — delete or fix (32 unique keys)

Keys that appear in both `references.bib` and a `refs/ch-*.bib` file must be
removed from **both** places; cross-references in `.qmd` files will then break
and will need to be removed or replaced.

| # | bibkey | Found in | Problem |
|---|--------|----------|---------|
| 1 | `kraft2021democratizing` | references.bib (p1), ch-17 | DOI resolves to "Are disagreements agreeable?" (JFE 2021); no Kraft/Chen/Dong/Sauermann "Democratizing Credit" paper exists. |
| 2 | `ooms2013lookahead` | references.bib (p1), ch-03 | DOI 10.1016/j.jempfin.2017.02.004 → 404; no such paper anywhere. |
| 3 | `navas2022optbinning` | references.bib (p1), ch-03 | DOI 10.1287/ijoc.2021.1151 is a vehicle-routing paper; only arXiv 2001.08025 is real (no INFORMS JoC publication). |
| 4 | `bequet2018benchmarking` | references.bib (p2), ch-16 | JMLR paper with this title/authors does not exist. |
| 5 | `wu2022sdcn` | references.bib (p2), ch-14 | Claims SIGKDD 2022; real paper is IEEE IoT Journal with different authors. |
| 6 | `goh2023fair` | references.bib (p2), appx-C | No Goh/Wang MNSC paper; real MS fairness paper is Hurlin/Pérignon/Saurin. |
| 7 | `dowd2014empirical` | references.bib (p2) | DOI 10.1017/asb.2013.26 → Guerreiro et al. "Open Bonus Malus" (wrong paper). |
| 8 | `gelbach2021testing` | references.bib (p3), ch-05 | No JFE 2021 paper by Gelbach/Hoberg/Strahan on disparate impact. |
| 9 | `hughes2022data` | references.bib (p3), ch-18 | DOI 10.1016/j.jbankfin.2021.106332 → "Exclamation Mark of Cain" (unrelated). |
| 10 | `ghosh2024cashflow` | references.bib (p3), ch-18 | Real paper is NBER w34148; entry cites w32354 which is "Macroeconomics of Mental Health". |
| 11 | `hurlin2022machine` | references.bib (p3), ch-05 | SSRN 2023 working paper only; never published in RCFS 2022. |
| 12 | `beran2019survival` | references.bib (p3), ch-09 | No "Survival and Cure Models in Retail Credit Risk" by Beran & Sobehart exists. |
| 13 | `chen2005sampling` | references.bib (p3), ch-10 | Title + coauthor "Åstrom" wrong; Chen's real coauthor is Thomas Åstebro with different titles. |
| 14 | `matthies2021reading` | references.bib (p4), ch-25 | DOI redirects to unrelated paper; Matthies works on asset pricing, not loan text. |
| 15 | `tjoa2023llmfin` | references.bib (p4), ch-26 | Not in ICAIF'23 proceedings; DOI does not resolve. |
| 16 | `vallee2020boom` | references.bib (p4), ch-20 | Real Vallée "Call Me Maybe?" is subtitled "Exercising Contingent Capital", different topic + venue; cited DOI is a different paper. |
| 17 | `lee2020linguistic` | references.bib (p4), ch-25 | DOI 404; title matches 2014 Hau JIMF paper, not Lee/Naranjo/Sirmans. |
| 18 | `aghaei2024llm` | references.bib (p5), ch-26 | No Aghaei/Azizi/Vayanos MS credit-risk paper; their real work is fair decision trees. |
| 19 | `reynolds2009colombia` | references.bib (p5), ch-31 | DOI → Jarrow & Xu "Credit Rating Accuracy and Incentives" (unrelated). |
| 20 | `burlando2023digital` | references.bib (p5), ch-31 | DOI → Gindling & Ronconi "Minimum wage policy and inequality in Latin America". |
| 21 | `brailovskaya2024consumer` | references.bib (p5) | DOI → "Conservative News Media and Criminal Justice"; real Brailovskaya/Dupas/Robinson EJ 2024 paper has different DOI/issue/pages. |
| 22 | `mckinney2023climate` | references.bib (p6), ch-33 | DOI → "How ETFs Amplify the Global Financial Cycle"; author list unfindable. |
| 23 | `breeden2024long` | references.bib (p6), ch-33 | DOI → Kim/Cho/Ryu student-loan recovery paper. |
| 24 | `agarwal2020climate` | references.bib (p6), ch-33 | DOI → Bongaerts/Cremers/Goetzmann "Tiebreaker" (J. Finance 2012); wrong authors/year/volume. |
| 25 | `mariani2024quantum` | references.bib (p6), ch-33 | No paper by Mariani/Scarselli/Barzanti; the real quantum credit-scoring paper is Schetakis et al. |
| 26 | `alipourfard2024climate` | references.bib (p6), ch-33 | No AAAI-2024 paper by these authors on climate-transition credit risk. |
| 27 | `nguyen2021machine` | references.bib (p6) | DOI → Gu/Lv/Liu/Peng "Impact of Quantitative Easing on Cryptocurrency". |
| 28 | `nguyen2022vietnamcredit` | references.bib (p6) | DOI → Lanchimba et al. "Franchising and country development" (unrelated). |
| 29 | `ross2014feature` | references.bib (p6), ch-34 | DOI → "Analytic Approach to Quantify Sensitivity of CreditRisk+" (unrelated); Rossi/Turrini "Model Validation in Banking" unfindable. |
| 30 | `breedenbarakova2017multidim` | ch-32 | DOI → Correlation/CVA paper; no Breeden/Barakova scorecard-recalibration paper at that DOI. |
| 31 | `hwang2023kakaobank` | references.bib (p5), ch-31 | DOI 404; no paper findable in "Korea and the World Economy". |
| 32 | `chernozhukov2024long` | references.bib (p6), ch-33 | JASA DOI 404; paper exists only as arXiv, not published in JASA 119(547). |

---

## UNCERTAIN — real paper, but bib metadata wrong (42 unique keys)

Mostly: wrong DOI, wrong year/volume/pages, incomplete author list, or misleading
bibkey. Paper is **real** — fix the metadata rather than delete.

### DOI or venue is wrong (paper exists elsewhere)
| bibkey | Fix needed |
|--------|-----------|
| `agarwal2020fintech` | DOI 10.1287/mnsc.2023.4878 is wrong paper; confirm MNSC placement. |
| `hurlin2024fairness` | Wrong DOI; correct is 10.1287/mnsc.2022.03888 (MS vol 72). |
| `hau2021fintech` | Real paper has 6 authors (add Lin, Wei). |
| `hand2002choice` | DOI may differ from canonical `10.1080/02664760050076371`. |
| `gao2023evidence` | DOI 10.1093/rfs/hhaa034 is wrong article; correct is likely hhaa046. |
| `balyuk2023reintermediation` | Actual JFQA vol 59(5), pp 1997–2037, 2024 — not 58(5) 2023. |
| `behn2016limits` | DOI typo: correct is 10.1111/jofi.13124 (not 13114). |
| `bumacov2014marketing` | DOI typo: 10.1002/jsc.1985 (not 1988). |
| `tran2022xgboost` | Published Springer LNCS 2021, vol 12709 — not 2022 vol 13343. |
| `vo2020vnstress` | Real paper is Cogent Econ & Finance 2018, not 2020. |
| `leboulanger2021tet` | DOI 404; bibkey author mismatch. |
| `ambrose2001prepayment` | Real paper is in Real Estate Economics 29(2) 2001, not RFS; DOI 404. |
| `beutel2019putting` | Coauthor list in bib does not match published authors. |
| `babaev2022coles` | SIGMOD 2022 is real; author list in bib is wrong. |
| `kvamme2018predicting` | Coauthor "Sj{\\"o}ursen" is a typo for "Sjursen". |
| `hoffman1983interpretation` | 1979 Yale L.J. Note is unsigned; attribution to Hoffmann uncertain; year mismatch. |
| `letizia2022supplychain` | Real paper is EPJ Data Science 2019; bibkey year + topic wrong. |
| `khemakhem2018credit` | Real year 2015, not 2018. |
| `ghodsi2023decision` | Minor DOI digit difference; paper is real (ICCV 2023). |
| `larsen2021news` | Real year 2019 (JoE), not 2021. |

### Bibkey year/title mismatch only (paper real, just rename key)
`veldkamp2023valuing` (NBER 28427 = 2021), `farboodi2023data` (dup of previous),
`jagtiani2019roles` / `jagtiani2019roles_alt` (duplicates),
`breeden2019survey` (year is 2020), `gelman2017prior` (AoAS 2008),
`foster1989resolving` (JASA 2004), `chen2023nonbank` (RFS 2019),
`rona2021big` (SER 2020), `miller2017impact` (MS 2018),
`goldfarb2023regulation` (MS 2011), `lowry2020initial` (FnT 2017),
`firth2021readability` (MS 2015), `gabaix2024llm` (NBER w32001, not w32768),
`liu2021ptuning` (AI Open 2024), `loukas2023edgar` (Banking77, not EDGAR),
`hamilton2024graphnn` (GeniePath AAAI 2019), `bauguess2020role` (SEC 2017),
`gordy2010small` (real 2013 IJCB; pages 33–71 not 38–77),
`kraus2013fairness` (real 2017 DSS paper), `navas2016optimal` (arXiv 2020),
`hue2023xper` (year says 2025; arXiv 2022), `angelino2018learning` (JMLR 2017/2018),
`chen2019howvaluable` (duplicate of `chen2023nonbank`).

### Could not confirm (paywalled / obscure venue)
`korablev2009validation`, `feelders2000credit`, `crosbie2003modeling`,
`alonso2020emfintech`, `agarwal2020bandit`, `kuck2003identifying`,
`breeden2012vintage`, `breeden2024xai`, `armstrong2013flow`, `chu2024vietnam`,
`assef2024realtime`.

---

## Recommended action

1. Delete the 32 **LIKELY-FABRICATED** entries from `references.bib` and from
   each `refs/ch-*.bib` listed above.
2. Grep for the deleted keys in `book/chapters/*.qmd` and `book/index.qmd`; for
   every hit, remove the citation and the sentence/paragraph that depended on it
   (or substitute a verified reference).
3. For the 42 **UNCERTAIN** entries in the "DOI or venue is wrong" subsection,
   fix the metadata (do not delete).
4. For bibkey-year mismatches, decide: rename key vs. leave as-is (low priority).

## Commands to locate usage of fabricated keys

```bash
# Run from repo root; lists every file/line that cites a fabricated key
for key in kraft2021democratizing ooms2013lookahead navas2022optbinning \
           bequet2018benchmarking wu2022sdcn goh2023fair dowd2014empirical \
           gelbach2021testing hughes2022data ghosh2024cashflow hurlin2022machine \
           beran2019survival chen2005sampling matthies2021reading tjoa2023llmfin \
           vallee2020boom lee2020linguistic aghaei2024llm reynolds2009colombia \
           burlando2023digital brailovskaya2024consumer mckinney2023climate \
           breeden2024long agarwal2020climate mariani2024quantum \
           alipourfard2024climate nguyen2021machine nguyen2022vietnamcredit \
           ross2014feature breedenbarakova2017multidim hwang2023kakaobank \
           chernozhukov2024long; do
  echo "=== $key ==="
  grep -rn "@$key" book/chapters book/index.qmd book/refs book/references.bib 2>/dev/null
done
```

---

## Usage map — where each fabricated key is cited

Only `.qmd` file citations require prose edits; `.bib` occurrences can be
deleted cleanly. Files in **bold** actually use the key as `@key` in prose:

| bibkey | .qmd files that cite it (prose edits needed) | .bib files to clean |
|--------|----------------------------------------------|---------------------|
| `kraft2021democratizing` | **17-digital-footprints** | ch-17, references |
| `ooms2013lookahead` | **03-data** | ch-03, references |
| `navas2022optbinning` | **03-data** | ch-03, references |
| `bequet2018benchmarking` | — | ch-16, references |
| `wu2022sdcn` | — | ch-14, references |
| `goh2023fair` | — | appx-C, references |
| `dowd2014empirical` | — | ch-16, references |
| `gelbach2021testing` | **05-regulation** | ch-05, references |
| `hughes2022data` | — | ch-18, references |
| `ghosh2024cashflow` | — | ch-18, references |
| `hurlin2022machine` | **05-regulation** | ch-05, references |
| `beran2019survival` | — | ch-09, references |
| `chen2005sampling` | — | ch-10, references |
| `matthies2021reading` | — | ch-25, references |
| `tjoa2023llmfin` | — | ch-26, references |
| `vallee2020boom` | — | ch-19, references |
| `lee2020linguistic` | — | ch-25, references |
| `aghaei2024llm` | — | ch-26, references |
| `reynolds2009colombia` | **11-trees-rules, 12-ensembles** | ch-31, references |
| `burlando2023digital` | **05-regulation, 31-inclusion-emerging** | ch-31, references |
| `brailovskaya2024consumer` | **07-logistic-scorecard, 31-inclusion-emerging** | ch-31, references |
| `mckinney2023climate` | **33-future** | ch-33, references |
| `breeden2024long` | **33-future** | ch-33, references |
| `agarwal2020climate` | — | ch-33, references |
| `mariani2024quantum` | **33-future** | ch-33, references |
| `alipourfard2024climate` | — | ch-33, references |
| `nguyen2021machine` | **11-trees-rules, 12-ensembles, 13-svm, 14-neural-networks, 15-imbalanced, 35-ifrs9-cecl-stress** | references, review_em_vietnam.md |
| `nguyen2022vietnamcredit` | **16-benchmarking, 19-p2p-lending, 20-bigtech-credit, 29-corporate-sme** | review_em_vietnam.md |
| `ross2014feature` | — | ch-34, references |
| `breedenbarakova2017multidim` | **32-dynamic-behavioral** | ch-32, references |
| `hwang2023kakaobank` | **31-inclusion-emerging** | ch-31, references |
| `chernozhukov2024long` | — | ch-33, references |

Prose-edit hotspots (descending impact):
- `nguyen2021machine` — cited in **6 chapters**; heaviest prose rewrite
- `nguyen2022vietnamcredit` — cited in **4 chapters**
- `brailovskaya2024consumer`, `burlando2023digital` — cited in 2 chapters each
- `reynolds2009colombia` — cited in 2 chapters

Note: `nguyen2022vietnamcredit` is referenced in chapter .qmd files but has no
entry in `references.bib` — only in `refs/review_em_vietnam.md` (a note file).
The citations in those four chapters are broken keys pointing at a fabricated
paper. Pick a real replacement or remove the sentences.
