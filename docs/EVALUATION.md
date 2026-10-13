# GMEE evaluation and human review

Engineering smoke tests and research experiments answer different questions.
Current implementation/validation is recorded in [Verification](VERIFICATION.md).
The research task, splits, ablations and uncertainty protocol are in
[Research](RESEARCH.md). No human-held-out accuracy result is claimed here.

## Task definitions

- Claim detection: distinguish check-worthy factual statements from opinions or questions. False factual statements remain claims. LIAR veracity labels cannot supervise claim extraction.
- Relatedness: SAME_STORY, EVOLVED, DISTINCT for a pair of claims about an event. A positive relatedness label does not establish mutation, copying or truth.
- Typed/contextual change: independent type and source-span annotations, plus PARAPHRASE, UPDATE_OR_CORRECTION, INCOMPATIBLE_ASSERTION, INSUFFICIENT_CONTEXT or UNRELATED.
- Evidence stance: evaluate the actual cited passage against the claim. Cosine similarity retrieves candidates; it never assigns entailment or contradiction.
- Lineage: evaluate links against independent attribution/event evidence. Publication order alone does not establish propagation.

## Authenticated annotation

Eval Lab requires an account. The server assigns the annotator identity from the
current authenticated user; clients cannot impersonate another reviewer or
upgrade an automatic vote to human. GET `/eval/rubric` supplies the versioned
instructions. GET `/eval/next` returns a blinded pair, without model scores or
other reviewers' decisions. POST `/eval/label` records the label, optional typed
changes, exact Unicode spans, contextual relation and notes. Revisions retain
history. An empty span list means explicitly reviewed with no changed spans;
omitting the field means that target was not reviewed.

Consensus requires at least two distinct authenticated humans and unanimous
current labels. Type, span and context consensus are reported separately;
incomplete annotations or dissent remain unresolved. The administrator's
independent review queue (`/eval/review/next`) presents disagreements without
exposing other votes; it never erases dissent or invents adjudicated gold.
A supervisor-led adjudication protocol remains a human research activity.

Automatic, legacy and test origins are separated from human gold. Weak labels
are diagnostics/training aids and cannot be promoted by majority vote into an
independent benchmark. Coverage sampling is appropriate for held-out work;
train-only uncertainty ordering must not adapt validation/test sampling.

## Frozen experiments

Use the admin split/export APIs and `backend/scripts/export_research_dataset.py`
to freeze records with event identity, human provenance, original claim/source
metadata, exact annotations and hashes. Shared events, claims, articles and
canonical text components stay in one split. Fit on train, select thresholds on
dev, and report untouched test comparisons with grouped uncertainty.

`backend/scripts/run_research_experiment.py` provides matched baselines and
ablations. Binary relatedness metrics cannot substitute for typed-change,
span, evidence or lineage evaluation. Review rare-class and event support;
passing the export's computational minimum is not a statistical guarantee.
Publish dataset/model/code revisions, raw predictions and negative findings.

## Legacy artifacts

Historical JSON/Markdown results are retained for traceability. The stored LIAR
claim-extraction score used the wrong target; the stored FEVER score compared
claims with other evaluation claims instead of Wikipedia evidence. Neither
validates the current task. `mutation_detection.json` contains zero pairs and
is unperformed. Perfect graph consistency from a three-claim SQLite fixture
is an engineering check. `backend/eval_publication_report.md` contains historical
weak-label diagnostics, not publication-grade human results.

The corrected runners in [evaluation](../evaluation/README.md) refuse LIAR as
an extraction target and require explicit frozen FEVER evidence passages. They
preserve earlier outputs by requiring a new result path. The current local
extraction model is FLAN-T5-base, retrieval is all-mpnet-base-v2 (768 dimensions),
and local stance is three-class DeBERTa. Record exact resolved revisions;
changing embedding models requires complete versioned re-embedding.

## Remaining scientific work

Recruit independent annotators, pilot the rubric, curate licensed event groups,
freeze held-out data, run matched baselines and ablations, measure type/span
and correction false-alarm errors, and report grouped uncertainty. A working
system or a successful synthetic smoke test does not establish paper novelty
or real-world factual accuracy.
