# GMEE publication plan

This plan supersedes the historical BGE-M3/LIAR extraction campaign. The deployed
retrieval model is all-mpnet-base-v2 (768 dimensions); FLAN-T5-base selects
source sentences and three-class DeBERTa supplies passage stance. Use actual
model/code/data revisions in a manuscript. A working application is not evidence
of novelty or benchmark accuracy.

## Focused contribution

Investigate whether quantity-, time- and provenance-aware claim-change analysis
reduces false mutation links caused by paraphrases and legitimate reporting
updates on unseen events. Treat unverified source references, inferred parents
and uncertain context explicitly. Do not claim observed copying or global
misinformation detection from semantic similarity.

## Submission requirements

1. Complete the current engineering gates and demonstrate a full attributable article → claim → change → cited passage → human review workflow. See [Verification](docs/VERIFICATION.md).
2. Compare contemporary temporal claim-change/correction systems and define the new method or resource contribution. Remove unsubstantiated “first system” and state-of-the-art claims.
3. Pilot [the annotation rubric](docs/EVALUATION.md), then collect independent human labels with event/source context, exact changed spans and unresolved cases. Machine votes are not gold.
4. Freeze event-disjoint train/dev/test data, preserving labels, revisions, licenses and content hashes. The research exporter rejects leakage; computational minimum counts do not guarantee statistical power.
5. Run matched embedding/lexical/pair baselines, quantity/time/provenance ablations and separate typed/contextual/span evaluations. Fit on train, select on dev, and use test once for final comparison. Report grouped uncertainty, support and errors.
6. Evaluate evidence retrieval/stance separately if making verification claims. LIAR truth labels do not measure claim extraction. FEVER requires actual evidence and its official scorer for an official FEVER-score claim.
7. Measure disclosed-hardware runtime, memory and cold/warm inference. Load, recovery and analyst usefulness need their own evidence.
8. Release permitted data or reconstructable acquisition instructions, frozen manifests, model/config/code revisions, predictions, exact commands and an accurate limitations discussion.

See [Research](docs/RESEARCH.md) for the experiment protocol and UNPERFORMED
ledger. Select a venue after the actual contribution/results are known. A
system demo, applied study or dataset paper may suit the final contribution;
acceptance and a “10/10” rating cannot be established by adding screens.
Historical result artifacts remain exploratory and must not be reported as
new independently annotated experiments.
