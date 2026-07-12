"""Community detection and LLM-powered community summarization."""

import json
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from .extraction import call_gemini


def detect_communities(G: nx.Graph) -> dict[int, list[str]]:
    """Detect communities using greedy modularity optimization.

    Similar to the Leiden algorithm used by Microsoft's GraphRAG,
    this groups densely connected entities into thematic clusters.

    Args:
        G: The knowledge graph.

    Returns:
        Dict mapping community IDs to lists of member entity names.
    """
    if len(G.nodes) == 0:
        return {}

    communities = greedy_modularity_communities(G)
    community_map = {}
    for i, comm in enumerate(communities):
        community_map[i] = sorted(list(comm))
        for node in comm:
            G.nodes[node]["community"] = i

    return community_map


def generate_community_summaries(
    community_map: dict, G: nx.Graph, api_key: str
) -> dict[int, str]:
    """Generate summaries for all communities in a single batched API call.

    Args:
        community_map: Dict mapping community IDs to member lists.
        G: The knowledge graph.
        api_key: Google Gemini API key.

    Returns:
        Dict mapping community IDs to summary strings.
    """
    all_info = []
    for cid, members in community_map.items():
        details = []
        for m in members:
            desc = G.nodes[m].get("description", "N/A")
            mtype = G.nodes[m].get("type", "unknown")
            details.append(f"  - {m} ({mtype}): {desc}")

        edges = []
        for m1 in members:
            for m2 in members:
                if G.has_edge(m1, m2):
                    descs = G[m1][m2].get("descriptions", ["related"])
                    edges.append(f"  {m1} -> {m2}: {descs[0]}")

        all_info.append(
            f"COMMUNITY {cid}:\nMembers:\n"
            + "\n".join(details)
            + "\nRelationships:\n"
            + ("\n".join(edges) if edges else "  (no internal edges)")
        )

    prompt = (
        "Summarize each community below in 2-3 sentences. "
        "Focus on the theme and key relationships.\n\n"
        + "\n\n".join(all_info)
        + "\n\nReturn a JSON object where keys are community IDs "
        "(as strings) and values are summary strings.\n"
        "Return ONLY valid JSON, no markdown."
    )

    raw = call_gemini(prompt, api_key)
    if raw:
        try:
            parsed = json.loads(raw)
            return {int(k): v for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: template-based summaries
    return {
        cid: f"Community of {len(members)} entities: {', '.join(members[:5])}"
        for cid, members in community_map.items()
    }
