# GMEE research runners

These tools support defined experimental tasks; stored historical metrics are
not fresh validation. Use the backend virtual environment and record model,
code and data revisions. Human benchmark work remains unperformed; see
[Evaluation](../docs/EVALUATION.md) and [Research](../docs/RESEARCH.md).

## Claim detection

LIAR veracity labels are rejected because false factual claims remain claims.
Use independently annotated sentence-level binary labels, or ClaimBuster's
check-worthiness target (a separate task). Nonempty extraction is a proxy
classifier, not a measure of atomic decomposition or span fidelity.

```bash
backend/.venv/bin/python evaluation/scripts/eval_claim_extraction.py \
  --dataset claimbuster --data-path evaluation/data/baselines/claimbuster_test.csv \
  --output evaluation/results/claimbuster-new-run.json --sample-size 500
```

The result path must be new. Confirm source licensing, label provenance and
frozen sampling before interpreting any result scientifically.

## FEVER supplied-evidence evaluation

Prepare separate frozen JSONL records `{id, passages:[{page,line,text}]}` for
every evaluated claim. An explicit empty passage list means missing evidence.
The evaluator never retrieves other evaluation claims as evidence.

```bash
backend/.venv/bin/python evaluation/scripts/eval_verdict_engine.py \
  --data evaluation/data/baselines/fever_dev.jsonl \
  --evidence /path/to/frozen-wikipedia-passages.jsonl --evidence-mode retrieved \
  --output evaluation/results/fever-new-run.json
```

`oracle` mode is an upper-bound supplied-evidence diagnostic; `retrieved` mode
requires retrieval independent of held-out labels. This runner measures passage
NLI aggregation, not the weighted corpus score or the official FEVER score.
Use the official evidence-set scorer before claiming the latter.

## Mutation and graph tools

The compatibility-named `eval_mutation_detection.py` is only a cosine-threshold
baseline on pair labels. It cannot validate typed changes, parent selection,
causal propagation or factual accuracy. Use the deployed 768-dimensional model
and independent data. The versioned backend research harness supports proper
frozen splits, fair baseline comparisons and separate typed-target evaluation.

`eval_graph_quality.py` checks structural consistency; its stored three-claim
fixture result is not measured real-world lineage precision. Empty-pair results
are UNPERFORMED. Preserve historical results and generate new named artifacts.

Dataset acquisition is implemented in `scripts/download_datasets.py` with an
acquisition manifest; redistribution rights must be checked separately.
