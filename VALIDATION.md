# Validation record

Validated locally on Python 3.12, 8 November 2026. Tests use synthetic data and mocked provider responses; no private documents or live paid API calls are required.

## Result

`python -m pytest -q`: **26 passed**. `pip check`: no broken requirements. `git diff --check`: clean.

## Automated coverage

- Exact source spans and stable chunk IDs; invalid overlap and empty inputs.
- PDF page metadata through chunking.
- Directed predicates, duplicate evidence handling and explicit alias mapping.
- Graph round-trip including community labels and provenance.
- Empty and edgeless graphs.
- Malformed extraction schemas, missing endpoints and invalid weights.
- Cache reuse, corrupt-cache recovery and resume after a failed partial build.
- Last good index preserved when a rebuild fails.
- Concurrent builds in two distinct workspaces stay isolated.
- Citation IDs outside retrieved context and uncited claims are rejected.
- Graph context and passage-budget bounds.
- Retrieval-only mode is not labeled as generated answers or answer accuracy.
- Authentication fails without retries or secret leakage; transient errors retry with usage recording.
- Provider failure retains retrieved evidence for inspection.
- Streamlit app simulation: initial render, comparison, rerun persistence, benchmark, reset and mocked live-answer judgment saving.
- Corrupt PDF/DOCX handling, model-change cache invalidation, bounded network timeout retries and no-evidence abstention.

## Measured retrieval fixture

Run `python evaluate.py --output examples/synthetic_retrieval_results.json` to reproduce the procedure. Latencies vary by machine. Five answerable questions contribute to source-recall metrics; one unanswerable question is included in detailed rows but excluded from those aggregates. The graph and lexical methods tie in this fixture. It is unsuitable for resume performance claims.

## Limits of this validation

Live Mistral extraction and generation were not tested because no provider credential was supplied to this environment. HTTP behavior was tested with mocked responses. Streamlit flows were tested through AppTest; the separate cloud browser could not reach this local server, so no rendered-browser visual pass is claimed. GitHub-hosted CI must run after the branch is pushed. Security review, load testing, independent answer judgments, dense-vector comparisons and real-document evaluations remain outstanding.
