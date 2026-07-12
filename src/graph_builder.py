"""Knowledge graph construction from extracted entities and relationships."""

import networkx as nx


def build_knowledge_graph(extractions: list[dict]) -> nx.Graph:
    """Build a NetworkX graph from LLM-extracted data.

    Merges duplicate entities and accumulates edge weights when
    the same relationship appears across multiple text chunks.

    Args:
        extractions: List of extraction dicts, each containing
                     'entities' and 'relationships' lists.

    Returns:
        A NetworkX Graph with typed, described nodes and weighted edges.
    """
    G = nx.Graph()
    entity_map = {}

    # Add entities as nodes
    for extraction in extractions:
        for entity in extraction.get("entities", []):
            name = entity["name"].upper().strip()
            if name not in entity_map:
                entity_map[name] = entity
                G.add_node(
                    name,
                    type=entity.get("type", "unknown"),
                    description=entity.get("description", ""),
                )

    # Add relationships as weighted edges
    for extraction in extractions:
        for rel in extraction.get("relationships", []):
            src = rel["source"].upper().strip()
            tgt = rel["target"].upper().strip()
            if src in G.nodes and tgt in G.nodes:
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] += rel.get("weight", 1)
                    G[src][tgt]["descriptions"].append(
                        rel.get("description", "")
                    )
                else:
                    G.add_edge(
                        src,
                        tgt,
                        weight=rel.get("weight", 1),
                        descriptions=[rel.get("description", "")],
                    )

    return G
