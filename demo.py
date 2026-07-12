"""
Demo script — shows graph overview and runs sample queries.
Usage: python demo.py
"""

import json
import os
import networkx as nx
from dotenv import load_dotenv


def print_graph_overview():
    """Display knowledge graph statistics."""
    stats = json.load(open("output/graph_stats.json", encoding="utf-8"))
    G = nx.read_gml("output/knowledge_graph.gml")

    print("=" * 60)
    print("  KNOWLEDGE GRAPH OVERVIEW")
    print("=" * 60)
    print(f"  Entities:      {stats['num_entities']}")
    print(f"  Relationships: {stats['num_relationships']}")
    print(f"  Communities:   {stats['num_communities']}")
    print()
    print("  Entity Types:")
    for t, count in sorted(stats["entity_types"].items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"    {t:15s} {count:3d} {bar}")
    print()

    degrees = sorted(G.degree(), key=lambda x: -x[1])[:10]
    print("  Most Connected Entities:")
    for name, deg in degrees:
        print(f"    {name:30s} {deg} connections")
    print("=" * 60)


def run_interactive_search():
    """Launch interactive search mode."""
    from src.search import local_search, global_search

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nSet GEMINI_API_KEY in .env to enable search.")
        return

    G = nx.read_gml("output/knowledge_graph.gml")
    summaries = json.load(open("output/community_summaries.json", encoding="utf-8"))
    summaries = {int(k): v for k, v in summaries.items()}

    chunks = []
    for f in sorted(os.listdir("data/documents")):
        if f.endswith(".txt"):
            chunks.append(open(os.path.join("data/documents", f), encoding="utf-8").read())

    print("\n--- Interactive Search (type 'quit' to exit) ---")
    print("  'local <query>'  -> entity-focused search")
    print("  'global <query>' -> thematic search\n")

    while True:
        user_input = input("Query > ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.lower().startswith("global "):
            query = user_input[7:]
            print("\n[Global Search]...\n")
            result = global_search(query, summaries, api_key)
        else:
            query = user_input.replace("local ", "", 1)
            print("\n[Local Search]...\n")
            result = local_search(query, G, summaries, chunks, api_key)

        print(result)
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    load_dotenv()
    print_graph_overview()
    run_interactive_search()
