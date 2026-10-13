# GMEE Historical Evaluation Report — Exploratory Weak Labels

> Historical artifact: the original multi-signal machine-label methodology does not provide independently annotated publication evidence. Identical embedding baselines and sample counts do not establish validity. Values below are preserved as recorded; use [the current evaluation protocol](../docs/EVALUATION.md) and frozen human-held-out experiments.

> Generated at: 2026-10-03T00:13:20.582894

---

## 1. Evaluation Dataset Summary

| Metric | Value |
|--------|-------|
| Total eval pairs | **1271** |
| Total annotations (labels) | **3826** |
| Consensus pairs (majority vote) | **1271** |
| Positive pairs (SAME_STORY + EVOLVED) | **200** (15.7%) |
| Negative pairs (DISTINCT) | **1071** (84.3%) |
| Annotator count | **8** |
| Annotation method | Multi-signal NLP (NLI + Entity + Lexical) |

### Pairs by Similarity Bucket

| Bucket | Cosine Range | Pairs |
|--------|-------------|-------|
| b95 | 0.95–1.00 | 24 |
| b85 | 0.85–0.95 | 121 |
| b75 | 0.75–0.85 | 121 |
| b60 | 0.60–0.75 | 206 |
| b40 | 0.40–0.60 | 399 |
| b00 | 0.00–0.40 | 400 |

### Annotator Label Distribution

| Annotator | SAME_STORY | EVOLVED | DISTINCT | Total |
|-----------|-----------|---------|----------|-------|
| cline-check | 0 | 0 | 1 | 1 |
| cline-check-2 | 0 | 1 | 0 | 1 |
| entity-annotator | 181 | 135 | 955 | 1271 |
| lexical-annotator | 207 | 55 | 1009 | 1271 |
| nli-annotator | 14 | 45 | 1212 | 1271 |
| tester1 | 1 | 0 | 0 | 1 |
| tester2 | 1 | 0 | 0 | 1 |
| ui-sweep | 0 | 0 | 9 | 9 |

## 2. Inter-Annotator Agreement

| Annotator Pair | Shared Pairs | Cohen's κ | Raw Agreement |
|---------------|-------------|-----------|---------------|
| cline-check vs entity-annotator | 1 | 1.000 | 100.0% |
| cline-check vs lexical-annotator | 1 | 1.000 | 100.0% |
| cline-check vs nli-annotator | 1 | 1.000 | 100.0% |
| cline-check vs tester1 | 1 | 0.000 | 0.0% |
| cline-check vs tester2 | 1 | 0.000 | 0.0% |
| cline-check vs ui-sweep | 1 | 1.000 | 100.0% |
| cline-check-2 vs entity-annotator | 1 | 0.000 | 0.0% |
| cline-check-2 vs lexical-annotator | 1 | 0.000 | 0.0% |
| cline-check-2 vs nli-annotator | 1 | 0.000 | 0.0% |
| entity-annotator vs lexical-annotator | 1271 | 0.675 | 87.8% |
| entity-annotator vs nli-annotator | 1271 | 0.120 | 75.5% |
| entity-annotator vs tester1 | 1 | 0.000 | 0.0% |
| entity-annotator vs tester2 | 1 | 0.000 | 0.0% |
| entity-annotator vs ui-sweep | 9 | 0.000 | 88.9% |
| lexical-annotator vs nli-annotator | 1271 | 0.160 | 79.9% |
| lexical-annotator vs tester1 | 1 | 0.000 | 0.0% |
| lexical-annotator vs tester2 | 1 | 0.000 | 0.0% |
| lexical-annotator vs ui-sweep | 9 | 1.000 | 100.0% |
| nli-annotator vs tester1 | 1 | 0.000 | 0.0% |
| nli-annotator vs tester2 | 1 | 0.000 | 0.0% |
| nli-annotator vs ui-sweep | 9 | 0.000 | 88.9% |
| tester1 vs tester2 | 1 | 1.000 | 100.0% |
| tester1 vs ui-sweep | 1 | 0.000 | 0.0% |
| tester2 vs ui-sweep | 1 | 0.000 | 0.0% |

> **Average κ = 0.290** — Fair agreement (Landis & Koch, 1977)

## 3. Retrieval Performance

| Method | AUROC | 95% CI | Best F1 | F1@0.60 | Brier | ECE |
|--------|-------|--------|---------|---------|-------|-----|
| sbert | 0.9715 | [0.96, 0.98] | 0.8322 | 0.5952 | 0.1793 | 0.2980 |
| tfidf | 0.9372 | [0.92, 0.95] | 0.7471 | 0.1991 | 0.0826 | 0.0908 |
| engine_embedding | 0.9715 | [0.96, 0.98] | 0.8322 | 0.5952 | 0.1793 | 0.2980 |

### Statistical Significance (McNemar @0.60)

- **engine_embedding_vs_tfidf**: χ²=16.771, p=0.0000 (✓ significant)
- **sbert_vs_tfidf**: χ²=16.771, p=0.0000 (✓ significant)
- **engine_embedding_vs_sbert**: χ²=0.000, p=1.0000 (✗ not significant)

## 4. Per-Bucket Agreement

| Bucket | Pairs | SAME_STORY | EVOLVED | DISTINCT | Positive Rate |
|--------|-------|-----------|---------|----------|---------------|
| b95 | 24 | 21 | 0 | 3 | 87.5% |
| b85 | 121 | 95 | 6 | 20 | 83.5% |
| b75 | 121 | 50 | 13 | 58 | 52.1% |
| b60 | 206 | 0 | 15 | 191 | 7.3% |
| b40 | 399 | 0 | 0 | 399 | 0.0% |
| b00 | 400 | 0 | 0 | 400 | 0.0% |

## 5. Threshold Sweep

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.40 | 0.2296 | 1.0000 | 0.3735 |
| 0.45 | 0.3210 | 1.0000 | 0.4860 |
| 0.50 | 0.3868 | 1.0000 | 0.5579 |
| 0.55 | 0.4149 | 1.0000 | 0.5865 |
| 0.60 | 0.4237 | 1.0000 | 0.5952 |
| 0.65 | 0.5361 | 0.9650 | 0.6893 |
| 0.70 | 0.6254 | 0.9350 | 0.7495 |
| 0.75 | 0.6955 | 0.9250 | 0.7940 |
| 0.80 | 0.7752 | 0.8450 | 0.8086 |
| 0.85 | 0.8414 | 0.6100 | 0.7072 |
| 0.90 | 0.8889 | 0.2800 | 0.4259 |

## 6. Calibration (Reliability Diagram Data)

| Bin | N | Mean Score | Empirical Rate | Gap |
|-----|---|-----------|----------------|-----|
| [0.0, 0.1] | 257 | 0.0504 | 0.0000 | 0.0504 |
| [0.1, 0.2] | 107 | 0.1384 | 0.0000 | 0.1384 |
| [0.2, 0.3] | 31 | 0.2411 | 0.0000 | 0.2411 |
| [0.3, 0.4] | 5 | 0.3523 | 0.0000 | 0.3523 |
| [0.4, 0.5] | 354 | 0.4349 | 0.0000 | 0.4349 |
| [0.5, 0.6] | 45 | 0.5350 | 0.0000 | 0.5350 |
| [0.6, 0.7] | 173 | 0.6406 | 0.0751 | 0.5655 |
| [0.7, 0.8] | 81 | 0.7547 | 0.2222 | 0.5325 |
| [0.8, 0.9] | 155 | 0.8548 | 0.7290 | 0.1258 |
| [0.9, 1.0] | 63 | 0.9411 | 0.8889 | 0.0522 |

> ✅ **Sample size meets the minimum threshold for statistical validity.**
