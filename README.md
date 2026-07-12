# GraphRAG Engine

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Google_Gemini-API-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Analysis-orange)](https://networkx.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A from-scratch implementation of Microsoft's GraphRAG architecture** — extracts entities & relationships from documents using LLMs, builds knowledge graphs, detects communities, and enables intelligent Local & Global search. Includes an interactive Streamlit dashboard.

<p align="center">
  <img src="https://img.shields.io/badge/Entities-119-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Relationships-127-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Communities-15-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/API_Calls-~5-orange?style=for-the-badge" />
</p>

---

## What is GraphRAG?

Traditional RAG retrieves text chunks via vector similarity — it finds passages that *look similar* to the query. **GraphRAG** goes further:

| | Traditional RAG | GraphRAG |
|---|---|---|
| **Retrieval** | Vector similarity on text chunks | Knowledge graph traversal |
| **Context** | Isolated passages | Entity neighborhoods + community summaries |
| **Strength** | Specific factual questions | Thematic, multi-hop reasoning |
| **Weakness** | Misses cross-document connections | More complex pipeline |

**Result**: GraphRAG outperforms traditional RAG by **70-80%** on thematic questions ([Microsoft Research, 2024](https://arxiv.org/abs/2404.16130)).

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌────────────────────┐
│  Documents   │───▶│   Chunking   │───▶│  LLM Entity        │
│  (.txt)      │    │  (500 words) │    │  Extraction        │
└──────────────┘    └──────────────┘    └────────┬───────────┘
                                                 │
                                                 ▼
┌──────────────┐    ┌──────────────┐    ┌────────────────────┐
│  Search      │◀───│  Community   │◀───│  Knowledge Graph   │
│  Local/Global│    │  Detection   │    │  (NetworkX)        │
└──────────────┘    └──────────────┘    └────────────────────┘
```

| Stage | Module | What Happens |
|-------|--------|-------------|
| Chunking | `src/chunking.py` | Split documents into overlapping segments |
| Extraction | `src/extraction.py` | LLM extracts entities + relationships as JSON |
| Graph Build | `src/graph_builder.py` | Entities → nodes, relationships → weighted edges |
| Communities | `src/community.py` | Detect clusters via greedy modularity optimization |
| Search | `src/search.py` | Local (entity neighborhood) or Global (community synthesis) |

---

## Quick Start

```bash
# Clone
git clone https://github.com/NikshayGoyal/GraphRAG-Engine.git
cd GraphRAG-Engine

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the indexing pipeline
python pipeline.py

# Launch the interactive dashboard
streamlit run app.py

# Or use the CLI demo
python demo.py
```

---

## Interactive Dashboard

**🔗 Live Demo: [graphrag-engine-nikshay.streamlit.app](https://graphrag-engine-nikshay.streamlit.app)**

Launch locally with `streamlit run app.py` to explore:

- **Knowledge Graph** — Interactive force-directed visualization with community filtering
- **Search** — Query the graph using Local (entity-focused) or Global (thematic) search
- **Communities** — Explore all 15 detected thematic clusters
- **Statistics** — Entity type distribution, top connected entities, community sizes

---

## Results

| Metric | Value |
|--------|-------|
| Documents Processed | 3 |
| Entities Extracted | **119** |
| Relationships Found | **127** |
| Communities Detected | **15** |
| Total API Calls | **~5** (vs 1000+ for Microsoft's GraphRAG) |

### Communities Discovered

| # | Theme | Key Members |
|---|-------|------------|
| 0 | Python Data Science Stack | Python, NumPy, Pandas, Scikit-learn, Snowflake |
| 1 | Classical Machine Learning | Arthur Samuel, SVM, Random Forest, IBM |
| 2 | Deep Learning Revolution | Geoffrey Hinton, AlexNet, Turing Award |
| 3 | GraphRAG & Knowledge Graphs | Microsoft Research, Leiden Algorithm |
| 4 | NLP & Transformers | BERT, GPT, Transformer, Google, OpenAI |
| 6 | Data Engineering | Apache Spark, Kafka, Airflow |
| 8 | RAG Systems | Pinecone, Weaviate, ChromaDB, LLMs |

---

## Project Structure

```
GraphRAG-Engine/
├── app.py                    # Streamlit interactive dashboard
├── pipeline.py               # Main indexing pipeline
├── demo.py                   # CLI demo with search
├── requirements.txt
├── LICENSE
│
├── src/                      # Modular source code
│   ├── __init__.py           # Package exports
│   ├── chunking.py           # Document loading & text chunking
│   ├── extraction.py         # LLM entity/relationship extraction
│   ├── graph_builder.py      # Knowledge graph construction
│   ├── community.py          # Community detection & summarization
│   └── search.py             # Local & Global search engines
│
├── data/
│   └── documents/            # Input text files
│       ├── 01_machine_learning.txt
│       ├── 02_data_science_tools.txt
│       └── 03_real_world_applications.txt
│
└── output/                   # Generated artifacts
    ├── extractions.json      # Raw entity/relationship data
    ├── knowledge_graph.gml   # Graph (importable in NetworkX/Gephi)
    ├── community_summaries.json
    └── graph_stats.json
```

---

## Comparison with Microsoft GraphRAG

| Feature | Microsoft GraphRAG | This Implementation |
|---------|-------------------|---------------------|
| Entity Extraction | LLM + gleanings | LLM (Gemini) |
| Knowledge Graph | NetworkX | NetworkX |
| Community Detection | Leiden Algorithm | Greedy Modularity |
| Community Reports | LLM-generated | LLM-generated (batched) |
| Local Search | ✅ | ✅ |
| Global Search | Map-Reduce | Community synthesis |
| Interactive Dashboard | ❌ | ✅ Streamlit |
| API Calls Needed | ~1000+ | **~5** |
| Dependencies | 90+ packages | 6 packages |

---

## Key Learnings

- **Graph vs Vector**: Knowledge graphs capture multi-hop relationships that vector stores miss
- **Community Detection**: Graph algorithms reveal thematic clusters automatically
- **Two Search Paradigms**: Local for specific questions, Global for thematic analysis
- **Token Efficiency**: Batched prompts reduce API costs by **200x**
- **Production Trade-offs**: Retry logic and rate limiting are critical for LLM systems

---

## References

- [From Local to Global: A Graph RAG Approach (Microsoft Research, 2024)](https://arxiv.org/abs/2404.16130)
- [Microsoft GraphRAG Repository](https://github.com/microsoft/graphrag)

## Author

**Nikshay Goyal** — Built from first principles to deeply understand GraphRAG architecture.

## License

[MIT](LICENSE)
