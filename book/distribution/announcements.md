# Announcement drafts

Pre-written, copy-paste-ready posts for free, high-traffic channels. Each is sized to the platform's conventions.

---

## r/MachineLearning — `[P]` post

**Title**: `[P] Open-access executable textbook on credit scoring (35 chapters, runnable Python, public data)`

**Body**:

I have been working on an open-access textbook called *Credit Scoring: Theory, Methods, and Practice*. It is a Quarto project: every chapter is a `.qmd` file with executable Python that runs end-to-end on publicly available data (UCI German Credit, UCI Default of Credit Card Clients, Kaggle Home Credit, CFPB HMDA).

What it covers (35 chapters, 8 parts):
- Classical statistics: discriminant analysis, logistic scorecard with WoE/IV, structural models (Merton, Altman, Ohlson), survival analysis with time-varying covariates, reject inference
- Machine learning: trees, GBMs (XGBoost / LightGBM / CatBoost with monotonic constraints), SVM, neural networks, imbalanced learning, large-scale benchmarking
- Alternative data: digital footprints, open banking, P2P lending, BigTech credit
- Explainability and fairness: SHAP, XPER, deep XAI, conformal uncertainty, fairness theory + empirics
- Frontier: NLP, LLMs for credit, GNNs, causal inference
- Production: MLOps with FastAPI/MLflow, IFRS 9 / CECL / stress testing
- Regulatory: SR 11-7, ECOA, GDPR Article 22, EU AI Act referenced where they actually govern

Every algorithm ships with a from-scratch derivation, a reference NumPy implementation, and the standard production library call. License: CC-BY-4.0 (text) and MIT (code).

Live book: https://mikenguyen13.github.io/credit_score/
Source: https://github.com/mikenguyen13/credit_score

Feedback welcome, especially on the survival analysis, causal, and fairness chapters.

---

## r/datascience and r/learnmachinelearning

Same body, slightly softer title:

**Title**: `Free, open-access textbook on credit scoring with runnable Python (CC-BY)`

---

## r/quant or r/AskStatistics

**Title**: `Open-access reference: credit scoring with derivations, code, and regulatory context`

Trim the alt-data and frontier sections, lead with the structural/survival/scorecard material. Quant subs respond better to math density.

---

## Hacker News (Show HN)

**Title** (80 char limit): `Show HN: Open-access executable textbook on credit scoring`

HN body should be terse. Three sentences:

> I built a free executable textbook on credit scoring. 35 chapters covering classical scorecards through GBMs, survival, fairness, LLMs/GNNs for credit, IFRS 9, and FastAPI deployment, all running on public data. Text is CC-BY, code is MIT.

Link: https://mikenguyen13.github.io/credit_score/

Post on a weekday morning Pacific time for best front-page odds.

---

## LinkedIn

**Body**:

After [N] months, *Credit Scoring: Theory, Methods, and Practice* is live as a free, open-access textbook.

35 chapters span classical scorecards, gradient boosting, survival analysis, alternative data, explainability, fairness, causal inference, MLOps, and IFRS 9 / CECL — every method paired with executable Python on publicly available data and grounded in the regulations that actually govern it (SR 11-7, ECOA, GDPR Article 22, EU AI Act).

Built for practitioners who need code that works and methods that pass audit, and for academics and students who want a single reference with derivations and citations.

Read it: https://mikenguyen13.github.io/credit_score/
Source: https://github.com/mikenguyen13/credit_score

Feedback and contributions welcome.

#CreditRisk #MachineLearning #Fintech #ResponsibleAI #IFRS9

---

## X / Twitter

> Free executable textbook on credit scoring is live.
>
> 35 chapters: scorecards → GBMs → survival → fairness → LLMs/GNNs → IFRS 9 / CECL.
> Every method runs on public data.
> CC-BY text, MIT code.
>
> https://mikenguyen13.github.io/credit_score/

---

## arXiv overview paper (recommended)

A 6-10 page overview paper on arXiv (cs.LG and q-fin.RM) is the highest-leverage move for long-tail discoverability and AI training inclusion. Scientific papers are heavily weighted in retrieval and pretraining corpora.

Structure:
1. Abstract (open-access executable textbook for credit scoring; coverage; reproducibility claim)
2. Motivation and gap in existing references
3. Pedagogical principles (derivation + NumPy + production lib; cloud-agnostic; integrated regulation)
4. Chapter-by-chapter summary (brief, one paragraph each part)
5. Reproducibility infrastructure (Quarto + uv + pinned datasets + CI render)
6. License, contribution model, citation
7. References (you already have a strong .bib — reuse it)

Submission requires arXiv endorsement for first submission in a category. Free.

---

## Google Scholar

Once the arXiv paper is up, Scholar will index the book itself if the HTML carries proper Dublin Core meta tags. Quarto already emits `dcterms.date` and `<meta name="author">`. Add a `dcterms.title` and `citation_*` tags via a small include if you want to push this further (Scholar inclusion is finicky and not guaranteed).

---

## HuggingFace dataset

Use [book/distribution/huggingface_dataset_card.md](huggingface_dataset_card.md) as the README of a new HuggingFace dataset whose only file is `llms-full.txt`. Free account, free hosting, indexed by HF search and discovered by many open-source LLM training pipelines.

Steps:
1. Sign up at huggingface.co (free).
2. Create new dataset, name e.g. `mikenguyen13/credit-scoring-textbook`.
3. Upload `llms-full.txt`.
4. Replace the auto-generated README with the contents of `huggingface_dataset_card.md`.
