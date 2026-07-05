# GraphRAG-Engine

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Google_Gemini-API-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Analysis-orange)](https://networkx.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A from-scratch implementation of Microsoft's GraphRAG architecture** — extracts entities & relationships from documents using LLMs, builds knowledge graphs, detects communities, and enables intelligent Local & Global search.

<p align="center">
  <img src="https://img.shields.io/badge/Entities-119-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Relationships-127-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Communities-15-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/API_Calls-~5-orange?style=for-the-badge" />
</p>

---

## What is GraphRAG?

Traditional RAG retrieves text chunks via vector similarity — it finds passages that *look similar* to the query. **GraphRAG** goes further:

```
                    Traditional RAG                         GraphRAG
                    ─────────────                          ────────
    Query → Vector Search → Top-K Chunks → LLM    Query → Knowledge Graph → Community Context → LLM
                                                          ↑ Entities + Relationships
                                                          ↑ Community Detection
                                                          ↑ Hierarchical Summaries
```

**Result**: GraphRAG outperforms traditional RAG by **70-80%** on thematic questions (Microsoft Research, 2024).

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

| Stage | What Happens | Tech Used |
|-------|-------------|-----------|
| Chunking | Split documents into overlapping segments | Python |
| Extraction | LLM extracts entities + relationships as structured JSON | Gemini API |
| Graph Build | Entities → nodes, relationships → weighted edges | NetworkX |
| Communities | Detect clusters of related entities | Greedy Modularity |
| Search | Local (entity neighborhood) or Global (community synthesis) | Gemini API |

---

## Results

### Knowledge Graph Statistics

| Metric | Value |
|--------|-------|
| Documents Processed | 3 |
| Entities Extracted | **119** |
| Relationships Found | **127** |
| Communities Detected | **15** |
| Total API Calls | **~5** (vs 1000+ for full GraphRAG) |

### Entity Distribution

| Type | Count | Examples |
|------|-------|---------|
| Concept | 36 | Machine Learning, Deep Learning, NLP |
| Person | 26 | Geoffrey Hinton, Yann LeCun, Andrew Ng |
| Tool | 23 | Pandas, NumPy, Scikit-learn, Apache Spark |
| Algorithm | 14 | Random Forest, XGBoost, t-SNE, K-Means |
| Organization | 11 | Google, OpenAI, IBM, Microsoft Research |
| Event | 5 | 2018 Turing Award, Netflix Prize |
| Technology | 3 | Transformer, BERT, ResNet |
| Dataset | 1 | ImageNet |

### Communities Discovered

| # | Theme | Key Members |
|---|-------|------------|
| 0 | **Python Data Science Stack** | Python, NumPy, Pandas, Scikit-learn, Snowflake |
| 1 | **Classical ML** | Arthur Samuel, SVM, Random Forest, IBM |
| 2 | **Deep Learning Revolution** | Geoffrey Hinton, AlexNet, Turing Award |
| 3 | **GraphRAG & Knowledge Graphs** | Microsoft Research, Leiden Algorithm |
| 4 | **NLP & Transformers** | BERT, GPT, OpenAI, Google |
| 6 | **Data Engineering** | Apache Spark, Kafka, Airflow |
| 8 | **RAG Systems** | Pinecone, Weaviate, ChromaDB, LLMs |

---

## Quick Start

```bash
# Clone
git clone https://github.com/NikshayGoyal/GraphRAG-Engine.git
cd GraphRAG-Engine

# Install
pip install -r requirements.txt

# Configure
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the full pipeline
python graphrag_mini.py

# Or run the demo (shows graph stats + sample queries)
python demo.py
```

### Interactive Search

After the pipeline runs, you can query your knowledge graph:

```
Query > local Who created the Transformer?
[Local Search] → Finds ASHISH VASWANI, GOOGLE, 2017 and traverses connections

Query > global What are the main themes in AI?
[Global Search] → Synthesizes across all 15 communities for a thematic answer
```

### Visualization

Open `knowledge_graph.html` in your browser for an interactive force-directed graph:
- **Click** nodes to see entity details and connections
- **Scroll** to zoom in/out
- **Drag** nodes to rearrange
- **Filter** by community using the dropdown

---

## Comparison: This Implementation vs Microsoft GraphRAG

| Feature | Microsoft GraphRAG | This Implementation |
|---------|-------------------|---------------------|
| Entity Extraction | LLM + gleanings | LLM (Gemini) |
| Knowledge Graph | NetworkX | NetworkX |
| Community Detection | Leiden Algorithm | Greedy Modularity |
| Community Reports | LLM-generated | LLM-generated (batched) |
| Local Search | ✅ | ✅ |
| Global Search | Map-Reduce | Community synthesis |
| DRIFT Search | ✅ | ❌ |
| API Calls Needed | ~1000+ | **~5** |
| Setup Complexity | High | Minimal |
| Dependencies | 90+ packages | 3 packages |

---

## Project Structure

```
GraphRAG-Engine/
├── graphrag_mini.py          # Core pipeline — from-scratch implementation
├── demo.py                   # Demo script with sample queries
├── generate_viz.py           # Generates interactive HTML visualization
├── knowledge_graph.html      # Interactive graph visualization
├── requirements.txt
├── LICENSE
├── input/                    # Source documents
│   ├── 01_machine_learning.txt
│   ├── 02_data_science_tools.txt
│   └── 03_real_world_applications.txt
└── output/                   # Generated artifacts
    ├── extractions.json      # Raw entity/relationship data
    ├── knowledge_graph.gml   # Graph file (importable)
    ├── community_summaries.json
    └── graph_stats.json
```

---

## Key Learnings

- **Graph vs Vector**: Knowledge graphs capture multi-hop relationships that flat vector stores miss
- **Community Detection**: Graph algorithms reveal thematic clusters automatically — no manual labeling
- **Two Search Paradigms**: Local (entity neighborhood) for specific questions, Global (community synthesis) for thematic questions
- **Token Efficiency**: Smart batching and structured prompts reduce API costs by **200x**
- **Production Trade-offs**: Rate limits, model availability, and retry logic are critical for real-world LLM systems

---

## References

- [From Local to Global: A Graph RAG Approach (Microsoft Research, 2024)](https://arxiv.org/abs/2404.16130)
- [Microsoft GraphRAG Repository](https://github.com/microsoft/graphrag)
- [GraphRAG Blog Post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

---

## Author

**Nikshay Goyal** — Built as a deep-dive project to understand and implement GraphRAG architecture from first principles.

## License

[MIT](LICENSE)
