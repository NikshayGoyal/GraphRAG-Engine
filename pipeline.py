"""
GraphRAG Engine — Main Pipeline
Orchestrates the full GraphRAG workflow: load, chunk, extract, build, detect, summarize.

Usage:
    python pipeline.py                     # Run with default settings
    python pipeline.py --input data/documents --output output
"""

import os
import json
import time
import argparse
import networkx as nx
from dotenv import load_dotenv

from src import (
    load_documents,
    chunk_text,
    extract_entities_relationships,
    build_knowledge_graph,
    detect_communities,
    generate_community_summaries,
)


def run_pipeline(input_dir: str, output_dir: str, api_key: str):
    """Execute the complete GraphRAG indexing pipeline.

    Args:
        input_dir: Path to directory containing .txt documents.
        output_dir: Path to save generated artifacts.
        api_key: Google Gemini API key.

    Returns:
        Tuple of (graph, community_map, summaries, chunks).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  GraphRAG Engine - Knowledge Graph Pipeline")
    print("=" * 60)

    # Step 1: Load & Chunk
    print("\n[1/5] Loading documents...")
    docs = load_documents(input_dir)
    print(f"  Loaded {len(docs)} documents")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc["text"], chunk_size=500, overlap=50))
    print(f"  Created {len(all_chunks)} text chunks")

    # Step 2: Extract Entities & Relationships
    print("\n[2/5] Extracting entities & relationships...")
    all_extractions = []
    for i, chunk in enumerate(all_chunks):
        print(f"  Chunk {i+1}/{len(all_chunks)}...", end=" ")
        extraction = extract_entities_relationships(chunk, api_key)
        all_extractions.append(extraction)
        n_ent = len(extraction.get("entities", []))
        n_rel = len(extraction.get("relationships", []))
        print(f"OK ({n_ent} entities, {n_rel} relationships)")
        time.sleep(2)

    with open(os.path.join(output_dir, "extractions.json"), "w", encoding="utf-8") as f:
        json.dump(all_extractions, f, indent=2)

    # Step 3: Build Knowledge Graph
    print("\n[3/5] Building knowledge graph...")
    G = build_knowledge_graph(all_extractions)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    nx.write_gml(G, os.path.join(output_dir, "knowledge_graph.gml"))

    # Step 4: Detect Communities
    print("\n[4/5] Detecting communities...")
    community_map = detect_communities(G)
    print(f"  Found {len(community_map)} communities")
    for cid, members in community_map.items():
        print(f"    Community {cid}: {len(members)} members - {', '.join(members[:4])}...")

    # Step 5: Generate Community Summaries
    print("\n[5/5] Generating community summaries...")
    summaries = generate_community_summaries(community_map, G, api_key)
    for cid, summary in summaries.items():
        print(f"  Community {cid}: {summary[:80]}...")

    with open(os.path.join(output_dir, "community_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    # Save stats
    stats = {
        "num_documents": len(docs),
        "num_chunks": len(all_chunks),
        "num_entities": G.number_of_nodes(),
        "num_relationships": G.number_of_edges(),
        "num_communities": len(community_map),
        "communities": {str(k): v for k, v in community_map.items()},
        "entity_types": {},
    }
    for node in G.nodes:
        t = G.nodes[node].get("type", "unknown")
        stats["entity_types"][t] = stats["entity_types"].get(t, 0) + 1

    with open(os.path.join(output_dir, "graph_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Pipeline complete!")
    print(f"  {stats['num_entities']} entities | {stats['num_relationships']} relationships | {stats['num_communities']} communities")
    print(f"  Artifacts saved to: {output_dir}/")
    print("=" * 60)

    return G, community_map, summaries, all_chunks


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="GraphRAG Engine Pipeline")
    parser.add_argument("--input", default="data/documents", help="Input documents directory")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY in .env file")
        exit(1)

    run_pipeline(args.input, args.output, api_key)
