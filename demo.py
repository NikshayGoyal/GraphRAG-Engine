"""
Demo script — runs sample queries against the knowledge graph.
Shows both Local Search (entity-focused) and Global Search (thematic).
Usage: python demo.py
"""

import json
import os
import networkx as nx
from graphrag_mini import local_search, global_search

def load_artifacts():
    G = nx.read_gml(os.path.join("output", "knowledge_graph.gml"))
    summaries = json.load(open(os.path.join("output", "community_summaries.json"), encoding="utf-8"))
    summaries = {int(k): v for k, v in summaries.items()}
    
    chunks = []
    for f in sorted(os.listdir("input")):
        if f.endswith(".txt"):
            chunks.append(open(os.path.join("input", f), encoding="utf-8").read())
    
    return G, summaries, chunks

def print_graph_overview(G):
    stats = json.load(open(os.path.join("output", "graph_stats.json"), encoding="utf-8"))
    
    print("=" * 60)
    print("  KNOWLEDGE GRAPH OVERVIEW")
    print("=" * 60)
    print(f"  Entities:      {stats['num_entities']}")
    print(f"  Relationships: {stats['num_relationships']}")
    print(f"  Communities:   {stats['num_communities']}")
    print()
    print("  Entity Types:")
    for t, count in sorted(stats['entity_types'].items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"    {t:15s} {count:3d} {bar}")
    print()
    
    # Top connected entities
    degrees = sorted(G.degree(), key=lambda x: -x[1])[:10]
    print("  Most Connected Entities:")
    for name, deg in degrees:
        print(f"    {name:30s} {deg} connections")
    print("=" * 60)

def run_demo():
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY in .env")
        print("Demo will show graph stats only (no search).\n")
    
    G, summaries, chunks = load_artifacts()
    print_graph_overview(G)
    
    if not api_key:
        return
    
    demo_queries = [
        ("local", "Who created the Transformer architecture and what did it enable?"),
        ("global", "What are the major themes and trends in modern AI and data science?"),
    ]
    
    for search_type, query in demo_queries:
        print(f"\n{'='*60}")
        print(f"  [{search_type.upper()} SEARCH] {query}")
        print(f"{'='*60}\n")
        
        if search_type == "local":
            result = local_search(query, G, summaries, chunks, api_key)
        else:
            result = global_search(query, summaries, api_key)
        
        print(result)
        print()

if __name__ == "__main__":
    run_demo()
