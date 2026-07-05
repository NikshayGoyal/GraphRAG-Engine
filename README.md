# DS Knowledge Navigator — Mini GraphRAG Implementation

> A from-scratch implementation of Microsoft's GraphRAG architecture for Data Science knowledge discovery. Built with Python, Google Gemini API, NetworkX, and community detection algorithms.

## What is GraphRAG?

Traditional RAG (Retrieval-Augmented Generation) retrieves relevant text chunks via vector similarity search. **GraphRAG** goes further by:

1. **Extracting entities & relationships** from documents using LLMs
2. **Building a knowledge graph** connecting all discovered entities
3. **Detecting communities** using graph algorithms (Leiden/Louvain)
4. **Generating community summaries** for thematic understanding
5. **Enabling two search modes**:
   - **Local Search**: Entity-focused, traverses graph neighborhoods
   - **Global Search**: Thematic, synthesizes across community summaries

## Architecture

```
Documents → Chunking → LLM Entity Extraction → Knowledge Graph
                                                      ↓
                                              Community Detection
                                                      ↓
                                             Community Summaries
                                                      ↓
                                          Local Search | Global Search
```

## Results

| Metric | Value |
|--------|-------|
| Documents | 3 (Machine Learning, Data Science Tools, Real-World Applications) |
| Text Chunks | 3 |
| Entities Extracted | 119 |
| Relationships Found | 127 |
| Communities Detected | 15 |
| API Calls Used | ~5 (vs 1000+ for full GraphRAG) |

### Sample Communities Discovered

| # | Theme | Key Entities |
|---|-------|-------------|
| 0 | Data Science Ecosystem | Snowflake, BigQuery, Scikit-learn, Pandas, NumPy |
| 1 | Classical ML | Arthur Samuel, IBM, SVM, Random Forest, Decision Trees |
| 2 | Deep Learning Revolution | Geoffrey Hinton, Yann LeCun, AlexNet, Turing Award |
| 3 | GraphRAG & Knowledge Graphs | GraphRAG, Leiden Algorithm, Microsoft Research |
| 4 | NLP & Transformers | BERT, GPT, Transformer, Google, Ashish Vaswani |
| 6 | Data Engineering | Apache Spark, Kafka, Airflow, Matei Zaharia |
| 8 | RAG Systems | Pinecone, Weaviate, ChromaDB, LLMs |

## Tech Stack

- **Python 3.12** — Core language
- **Google Gemini API** — LLM for entity extraction & search
- **NetworkX** — Graph construction and analysis
- **Greedy Modularity** — Community detection algorithm
- **python-dotenv** — Environment management

## Project Structure

```
ds-graphrag/
├── graphrag_mini.py          # Main pipeline (from scratch!)
├── input/
│   ├── 01_machine_learning.txt
│   ├── 02_data_science_tools.txt
│   └── 03_real_world_applications.txt
├── output/
│   ├── extractions.json      # Raw entity/relationship data
│   ├── knowledge_graph.gml   # Graph file (importable)
│   ├── community_summaries.json
│   └── graph_stats.json
├── graphrag/                 # Microsoft's original repo (reference)
├── settings.yaml             # GraphRAG config
└── .env                      # API keys
```

## How to Run

```bash
# 1. Install dependencies
pip install google-genai networkx python-dotenv

# 2. Set your API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run the pipeline
python graphrag_mini.py

# 4. Interactive search
# After pipeline completes, type queries:
#   local transformer architecture    → entity-focused search
#   global what are the main themes   → thematic search
```

## Key Learnings

1. **Graph vs Vector**: Knowledge graphs capture relationships that flat vector stores miss
2. **Community Detection**: Leiden/Louvain algorithms reveal thematic clusters automatically
3. **Two Search Paradigms**: Local (entity neighborhood) vs Global (community synthesis)
4. **Token Efficiency**: Smart chunking and batched API calls reduce costs 200x
5. **Real-World Trade-offs**: Rate limits, model availability, and cost require production-grade retry logic

## Comparison: My Implementation vs Microsoft GraphRAG

| Feature | Microsoft GraphRAG | My Implementation |
|---------|-------------------|-------------------|
| Entity Extraction | LLM-powered with gleanings | LLM-powered (Gemini) |
| Knowledge Graph | NetworkX | NetworkX |
| Community Detection | Leiden Algorithm | Greedy Modularity |
| Community Reports | LLM-generated | LLM-generated (batched) |
| Local Search | Yes | Yes |
| Global Search | Map-Reduce over communities | Synthesis over summaries |
| DRIFT Search | Yes | Not implemented |
| API Calls | ~1000+ | ~5 |
| Setup Complexity | High (many dependencies) | Minimal |

## Author

Built as a deep-dive portfolio project to understand and demonstrate GraphRAG architecture for Data Science applications.

## References

- [Microsoft GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [Microsoft GraphRAG Repository](https://github.com/microsoft/graphrag)
- [From Local to Global: A Graph RAG Approach](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
