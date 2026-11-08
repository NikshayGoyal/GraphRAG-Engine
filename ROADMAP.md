# Next engineering milestones

The current change addresses correctness, isolation, traceability and a usable evaluation workflow. The steps below require additional implementation or independently checked data; they are not completed capabilities.

1. **Real corpus and reviewed evaluation.** Select 10–20 public reports with reuse rights, stable URLs, publication dates and file hashes. Author approximately 100 questions with exact supporting passages, expected claims and explicit unanswerable labels. Separate development and test questions before tuning. Avoid inferring ground truth from retrieval output.
2. **Semantic baselines.** Add a fixed embedding model, cached vectors and a dense retriever; then a keyword/vector hybrid. Compare against the existing lexical and graph methods with the same documents and generation settings. Report per-category results, including losses.
3. **Entity resolution.** Build a labeled alias-pair set. Evaluate merge precision and recall; explicitly handle organizations with similar names and preserve time-dependent identities. The current alias mapping requires manual review.
4. **Community synthesis.** Evaluate bounded LLM community reports with claim-level source provenance, hierarchical communities and map-reduce. Compare their cost and answer coverage against the current extractive reports.
5. **Operational service.** Add authentication, durable tenant storage, background jobs, rate limits, cost budgets and redacted tracing before offering live indexing to anonymous users. Measure concurrency behavior and deletion semantics.
6. **Regression gating.** Once an independent dataset exists, gate changes on answer correctness, citation support and unanswerable behavior as well as retrieval recall. Keep provider/model/prompt versions and raw run records.

## Recruiter demonstration

1. Explain that the bundled corpus is synthetic and the displayed retrieval is real.
2. Run the same question through the baseline and graph method.
3. Open the source passages and the retrieved relationship edges.
4. Run the benchmark and explain why an aggregate tie is not a success claim.
5. With a configured provider, generate claims, review source support, record a judgment and export the run.
6. Explain cache reuse, partial failures, workspace isolation and remaining limitations.

## Resume wording after this branch is adopted

These describe implemented features, not measured business impact:

- Built a GraphRAG evidence lab with directed knowledge graphs, weighted communities and bounded two-hop retrieval.
- Added source-linked citations, isolated workspaces and cached extraction with atomic recovery from failed indexing runs.
- Implemented baseline comparisons, retrieval evaluation and human answer judgments with exportable latency and token logs.

Do not claim better answer accuracy, production scale, 93% savings or 200× cost reduction without matched, reproducible measurements. Do not call the TF-IDF baseline dense-vector RAG.
