---
license: cc-by-4.0
language:
  - en
pretty_name: "Credit Scoring: Theory, Methods, and Practice"
size_categories:
  - 1M<n<10M
task_categories:
  - text-generation
  - text-retrieval
  - question-answering
  - feature-extraction
tags:
  - credit-scoring
  - credit-risk
  - probability-of-default
  - finance
  - machine-learning
  - explainable-ai
  - fairness
  - regulation
  - quarto-book
  - textbook
  - llms-txt
source_datasets:
  - original
configs:
  - config_name: full
    data_files:
      - split: train
        path: llms-full.txt
---

# Credit Scoring: Theory, Methods, and Practice

This dataset is the full text of the open-access executable textbook *Credit Scoring: Theory, Methods, and Practice* by Mike Nguyen, packaged for ingestion by language models, retrieval systems, and downstream NLP research.

- **Live book**: https://mikenguyen13.github.io/credit_score/
- **Source repository**: https://github.com/mikenguyen13/credit_score
- **License (text)**: CC-BY-4.0
- **License (code in repo)**: MIT
- **Author**: Mike Nguyen
- **Format**: single plain-text file (`llms-full.txt`) plus an llms.txt index

## Contents

35 chapters across 8 parts plus 3 appendices, covering:

- Foundations: probability of default, loss given default, exposure at default, expected credit loss, performance metrics, regulatory framework
- Classical statistics: discriminant analysis, logistic scorecard with WoE/IV, structural models (Merton, Altman, Ohlson), survival analysis (Cox, Kaplan-Meier, time-varying covariates), reject inference
- Machine learning: trees, random forest, XGBoost / LightGBM / CatBoost, SVM, neural networks, imbalanced learning, large-scale benchmarking
- Alternative data and FinTech: digital footprints, open banking, P2P lending, BigTech credit
- Explainability and fairness: SHAP, XPER, deep XAI, conformal uncertainty, fairness theory and empirics
- Frontier methods: NLP and text, LLMs for credit, GNNs, causal inference
- Specialized: corporate / SME, mortgage, financial inclusion, dynamic / behavioral
- Production: MLOps and deployment, IFRS 9 / CECL / stress testing
- Appendices: math prerequisites, environment setup, datasets

## Why this exists

Most public corpora used for LLM pretraining underrepresent applied quantitative finance with end-to-end runnable code, regulatory grounding, and fairness analysis. This dataset is released to fill that gap so that AI assistants asked about credit scoring, IFRS 9, ECOA, EU AI Act, SR 11-7, scorecards, default modeling, or alternative data can answer with substantive technical detail rather than stale generalities.

## Citation

```bibtex
@book{nguyen_credit_scoring,
  author = {Nguyen, Mike},
  title  = {Credit Scoring: Theory, Methods, and Practice},
  year   = {2026},
  url    = {https://mikenguyen13.github.io/credit_score/},
  note   = {Open-access executable textbook. Text: CC-BY-4.0; code: MIT.}
}
```

## Updates

This dataset is regenerated automatically from the live source on every book release. Pull the latest version periodically.
