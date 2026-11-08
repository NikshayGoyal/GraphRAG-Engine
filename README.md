# GraphRAG Engine — Evidence Lab

A GraphRAG-inspired document investigation and evaluation app built with Python, NetworkX, Mistral and Streamlit. Compare a lexical retriever with graph-assisted retrieval, inspect the exact supporting passages, and record human judgments.

**Status:** tested engineering prototype. No claim of production readiness, independent answer accuracy, or superiority over dense-vector RAG. The default demo is explicitly synthetic and works without an API key.

## Run locally

Python 3.11 or 3.12:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
streamlit run app.py
```

The app starts with six hand-authored fictional company documents and a matching hand-authored graph. Searches execute the real retrievers; the no-key mode does not generate answers. Nothing in this demo is represented as a result from a real company or an independent benchmark.

For live extraction and answer generation, configure `MISTRAL_API_KEY` server-side in `.env` or Streamlit secrets. Never commit it. `MISTRAL_MODEL` is configurable; pin an available model snapshot for reproducible experiments rather than a moving `latest` alias. Optional per-million input/output token prices in `.env.example` enable cost estimates based on returned usage, not API call counts. Failed-request billing is not estimated.

## What you can do

- Compare **TF-IDF baseline**, **Graph + text**, and **Community + text** under identical passage limits and source-character budgets.
- Inspect retrieved passages with document names, PDF page numbers, character offsets, stable IDs, and retrieval reasons.
- Explore directed graph relationships and their evidence; filter the graph by weighted communities.
- Optionally generate cited claims with Mistral. Invalid or missing citation IDs fail validation. Semantic support still requires human review.
- Record correctness, citation support and abstention judgments; export session runs and judgments to JSON.
- Run a deterministic synthetic retrieval benchmark and inspect per-question failures.
- Upload a separate collection, see indexing failures, retry using cached successful chunks, and retain the prior usable index after a failed rebuild.

## Architecture and deliberate trade-offs

1. **Ingestion:** TXT, Markdown, PDF and DOCX are parsed into source documents. Chunks preserve exact spans of extracted text. A PDF page is retained as a unit; DOCX has no invented page numbers. Scanned PDFs need external OCR.
2. **Extraction:** Mistral JSON is validated for entity fields, relationship fields, existing endpoints and finite positive weights. Errors remain explicit. Provider calls use bounded retries for transient failures and per-request network timeouts; authentication errors fail immediately.
3. **Evidence graph:** a `MultiDiGraph` preserves direction, predicates and source chunk IDs. Case and Unicode/whitespace normalization are conservative. Optional explicit alias mappings are supported in `build_index`; there is no automatic fuzzy merge. Edge weights count distinct supporting chunks, not confidence or independent documents. Overlap can repeat evidence.
4. **Communities:** weighted greedy modularity operates on an undirected projection. Labels are assigned before atomic snapshot persistence. Reports are **extractive relationship records**, replacing the previous all-communities-in-one-LLM-call summaries. This makes their provenance inspectable, but does not reproduce Microsoft's hierarchical LLM reports or map-reduce global search.
5. **Retrieval and generation:** TF-IDF uses unigram/bigram cosine similarity. Graph mode links query terms to entities, adds bounded two-hop context, and ranks passages. Community mode consults at most five reports and prioritizes their source passages. Generation sees the selected source passages, not the whole corpus. All methods share the same source-character budget; this is not a token-budget equality claim.

Graph mode uses up to four seeds, eight incident edges per expansion and 40 visited entities. These are baseline heuristics, not learned relevance scores. The graph panel displays at most 150 matching entities.

## Isolation, caching and recovery

Each Streamlit session owns a random temporary directory. Uploads are parsed in memory; supplied filenames never become filesystem paths. Building a collection does not delete a shared documents directory or replace another session's graph. Reset immediately clears that session's temporary workspace; normal cleanup follows session-object disposal, so exact cleanup timing after disconnection is not guaranteed.

The cache is local to that workspace and keyed by chunk text, extraction prompt version and configured model. Identical chunks reuse extraction, while graph and communities are recomputed from the current collection. This is **incremental extraction**, not a fully incremental community algorithm. For a moving model alias, clear the cache when the provider changes the model or use a snapshot model ID.

`index.json` is an atomic authoritative snapshot, published only after all chunks succeed. `last_attempt.json` retains partial failures. Do not run concurrent CLI builds into the same output directory; separate Streamlit sessions already use distinct paths. This prototype has no durable multi-user database or authentication layer.

Uploads are limited to 20 files in the app, 5 MB per file, 300 PDF pages, 500,000 extracted characters per document and 200 chunks per build. These bound demo workloads; they do not replace an authenticated service's quota or sandboxing controls. Keep live indexing private or restrict access at the hosting layer before exposing a paid provider key to anonymous visitors.

## Evaluation

```bash
python -m pytest -q
python evaluate.py --output evaluation_results.json
```

The synthetic fixture contains six questions: two factual, two cross-document, one thematic, and one unanswerable. Ground truth is manually specified source filenames. At a limit of three passages, it measures required-source recall and whether all required sources were retrieved; unanswerable questions are excluded from recall aggregates. It does **not** measure answer correctness or abstention because the benchmark does not call an LLM.

The measured development run is in `examples/synthetic_retrieval_results.json`. Its graph and TF-IDF methods tie on aggregate source recall. This is a smoke test with very short documents, not evidence of a general retrieval advantage. Latency includes fitting a TF-IDF index per request and excludes model calls; six local queries are too few for a performance claim.

For real evaluation, create an independently reviewed question set over a fixed public corpus with document hashes, required passages, expected claims and unanswerable cases. Keep development and held-out questions separate. Add dense-vector and keyword/vector hybrid baselines, hold generation settings constant, and review correctness, citation support, abstention, latency and total indexing/query cost. Report failures and uncertainty; never copy Microsoft paper metrics into this project's results.

## Command-line indexing

```bash
python pipeline.py --input data/documents --output output
python demo.py
```

CLI indexing requires a configured key. `demo.py` intentionally uses the synthetic fixture and no live model. The old GML/JSON outputs remain historical artifacts and are not loaded by the new app. Reindex original documents to create provenance-complete version-2 snapshots; provenance cannot be reconstructed honestly from old entity-only citations.

## Validation and limits

See [VALIDATION.md](VALIDATION.md) for checks and [ROADMAP.md](ROADMAP.md) for remaining work. CI runs tests on Python 3.11 and 3.12. `requirements-tested.txt` records the local Python 3.12 environment; use the bounded main requirements for general installation.

Known limits include lexical entity linking, no OCR, no dense-vector baseline, no automatic alias disambiguation, no hierarchical community summaries, no calibrated confidence scores, and no independent accuracy benchmark. Citation-ID validation proves source membership, not entailment. A complete extraction run is not a guarantee that every extracted relationship is correct.

## References

- Edge et al., [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130).
- Microsoft GraphRAG [local search](https://microsoft.github.io/graphrag/query/local_search/) and [global search](https://microsoft.github.io/graphrag/query/global_search/).
- NetworkX [weighted greedy modularity](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html).
- Mistral [JSON mode](https://docs.mistral.ai/inference/conversations/structured-output/json_mode).

MIT license. Original project by Nikshay Goyal.
