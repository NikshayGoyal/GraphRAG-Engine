"""Local and Global search engines for querying the knowledge graph."""

from google import genai
import networkx as nx


def local_search(
    query: str,
    G: nx.Graph,
    community_summaries: dict,
    chunks: list[str],
    api_key: str,
) -> str:
    """Entity-focused search that traverses graph neighborhoods.

    Finds entities matching the query via keyword overlap, gathers
    their neighbors and edge descriptions, then uses the LLM to
    synthesize an answer grounded in graph context.

    Args:
        query: User's natural language question.
        G: The knowledge graph.
        community_summaries: Dict of community summaries.
        chunks: Original text chunks for additional context.
        api_key: Google Gemini API key.

    Returns:
        LLM-generated answer string.
    """
    client = genai.Client(api_key=api_key)

    # Score entities by keyword overlap with query
    query_words = set(query.upper().split())
    scored = []
    for node in G.nodes:
        node_words = set(node.split())
        desc_words = set(G.nodes[node].get("description", "").upper().split())
        overlap = len(query_words & (node_words | desc_words))
        if overlap > 0:
            scored.append((node, overlap))

    scored.sort(key=lambda x: -x[1])
    top = [n[0] for n in scored[:10]]

    # Build context from entity neighborhoods
    context_parts = []
    for entity in top:
        desc = G.nodes[entity].get("description", "")
        neighbors = list(G.neighbors(entity))
        edge_info = []
        for n in neighbors[:5]:
            if G.has_edge(entity, n):
                descs = G[entity][n].get("descriptions", ["related"])
                edge_info.append(f"  -> {n}: {descs[0]}")
        context_parts.append(
            f"Entity: {entity}\nDesc: {desc}\nConnections:\n"
            + "\n".join(edge_info)
        )

    # Add relevant text chunks
    relevant_chunks = [
        c[:500] for c in chunks if any(e.lower() in c.lower() for e in top[:3])
    ]

    prompt = f"""Answer using ONLY the knowledge graph context and text below.

KNOWLEDGE GRAPH:
{chr(10).join(context_parts[:5])}

TEXT:
{chr(10).join(relevant_chunks[:2])}

QUESTION: {query}

Provide a detailed answer citing specific entities and relationships."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )
        return response.text
    except Exception as e:
        return f"**Error generating response:**\n{str(e)}\n\nPlease verify your GEMINI_API_KEY and try again."


def global_search(
    query: str, community_summaries: dict, api_key: str
) -> str:
    """Thematic search that synthesizes across community summaries.

    Unlike local search, global search provides high-level thematic
    answers by analyzing patterns across all detected communities.

    Args:
        query: User's natural language question.
        community_summaries: Dict of community summaries.
        api_key: Google Gemini API key.

    Returns:
        LLM-generated answer string.
    """
    client = genai.Client(api_key=api_key)

    summaries_text = "\n\n".join(
        [f"Community {cid}: {s}" for cid, s in community_summaries.items()]
    )

    prompt = f"""Answer using community summaries from a knowledge graph.

COMMUNITY SUMMARIES:
{summaries_text}

QUESTION: {query}

Synthesize across all relevant communities. Highlight themes and connections."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )
        return response.text
    except Exception as e:
        return f"**Error generating response:**\n{str(e)}\n\nPlease verify your GEMINI_API_KEY and try again."
