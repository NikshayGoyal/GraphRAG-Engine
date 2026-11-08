# Add source provenance, isolated indexing and a GraphRAG evaluation lab

Uploading documents previously replaced a shared directory, saved graphs omitted community labels, and answer citations could not be traced to original text. This change gives each Streamlit session an isolated workspace and preserves source spans through extraction, directed graph construction, retrieval and claim-level citation validation. Failed rebuilds retain the last good index and successful chunks can be reused on retry.

The dashboard now compares a TF-IDF baseline with bounded graph and community retrieval, displays source passages and latency, records human judgments, and exports run data. A clearly labeled synthetic demo works without provider access. Unsupported cost/quality claims are removed. Community reports are now deterministic extractive relationship records; hierarchical LLM summaries and dense-vector comparisons remain future work.

Validation: 26 local tests pass, including Streamlit AppTest flows, concurrent-workspace isolation, cache recovery, citation rejection and mocked provider retry/error behavior. Dependency checks and diff whitespace checks pass. The six-question synthetic fixture shows an aggregate tie between graph and lexical retrieval; no independent accuracy or cost improvement is claimed. Live Mistral calls and rendered-browser QA were not performed. CI is configured for Python 3.11/3.12 and must run after push.

Deployment: review this branch before merging into the Streamlit deployment branch. Configure the model key server-side only for live features. The no-key demo starts without legacy output files. Keep live paid indexing access-controlled; app-level authentication and durable tenant storage are outside this change.
