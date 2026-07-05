"""
DS Knowledge Navigator - A Mini GraphRAG Implementation
Built from scratch to demonstrate Graph-based Retrieval Augmented Generation.
Uses: Python, Google Gemini API (google.genai), NetworkX, Community Detection
"""

import os
import json
import re
import time
import networkx as nx
from pathlib import Path

# ============================================================
# STEP 1: Document Loading & Chunking
# ============================================================

def load_documents(input_dir):
    docs = []
    for f in sorted(Path(input_dir).glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        docs.append({"id": f.stem, "text": text, "source": f.name})
        print(f"  Loaded: {f.name} ({len(text)} chars)")
    return docs


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ============================================================
# STEP 2: Entity & Relationship Extraction (LLM-powered)
# ============================================================

EXTRACTION_PROMPT = """You are an expert at extracting structured knowledge from text.
Given the following text, extract ALL entities and relationships.

TEXT:
{text}

Return a valid JSON object with this EXACT structure:
{{
  "entities": [
    {{"name": "ENTITY_NAME", "type": "person|organization|technology|algorithm|concept|tool|dataset|event", "description": "one-line description"}}
  ],
  "relationships": [
    {{"source": "ENTITY_A", "target": "ENTITY_B", "description": "how A relates to B", "weight": 1}}
  ]
}}

Rules:
- Entity names should be UPPERCASE
- Extract ALL people, organizations, technologies, algorithms mentioned
- Include relationships like "created", "works_at", "is_part_of", "used_for", etc.
- Return ONLY valid JSON, no markdown fences
"""


def call_gemini(prompt, api_key, retries=3):
    """Call Gemini API with retry logic."""
    from google import genai

    client = genai.Client(api_key=api_key)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            return response.text
        except Exception as e:
            print(f"  [Retry {attempt+1}/{retries}] API error: {str(e)[:80]}")
            time.sleep(5 * (attempt + 1))

    return None


def extract_entities_relationships(text, api_key):
    """Use Gemini API to extract entities and relationships from text."""
    raw = call_gemini(EXTRACTION_PROMPT.format(text=text), api_key)
    if not raw:
        return {"entities": [], "relationships": []}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"entities": [], "relationships": []}


# ============================================================
# STEP 3: Knowledge Graph Construction
# ============================================================

def build_knowledge_graph(extractions):
    G = nx.Graph()
    entity_map = {}

    for extraction in extractions:
        for entity in extraction.get("entities", []):
            name = entity["name"].upper().strip()
            if name not in entity_map:
                entity_map[name] = entity
                G.add_node(name,
                          type=entity.get("type", "unknown"),
                          description=entity.get("description", ""))

    for extraction in extractions:
        for rel in extraction.get("relationships", []):
            src = rel["source"].upper().strip()
            tgt = rel["target"].upper().strip()
            if src in G.nodes and tgt in G.nodes:
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] += rel.get("weight", 1)
                    G[src][tgt]["descriptions"].append(rel.get("description", ""))
                else:
                    G.add_edge(src, tgt,
                              weight=rel.get("weight", 1),
                              descriptions=[rel.get("description", "")])

    return G


# ============================================================
# STEP 4: Community Detection
# ============================================================

def detect_communities(G):
    from networkx.algorithms.community import greedy_modularity_communities

    if len(G.nodes) == 0:
        return {}

    communities = greedy_modularity_communities(G)
    community_map = {}
    for i, comm in enumerate(communities):
        community_map[i] = sorted(list(comm))
        for node in comm:
            G.nodes[node]["community"] = i

    return community_map


def generate_community_summaries(community_map, G, api_key):
    """Generate summaries for all communities in ONE API call."""
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
            f"COMMUNITY {cid}:\nMembers:\n" + "\n".join(details) +
            "\nRelationships:\n" + ("\n".join(edges) if edges else "  (no internal edges)")
        )

    prompt = f"""Summarize each community below in 2-3 sentences. Focus on the theme and key relationships.

{chr(10).join(all_info)}

Return a JSON object where keys are community IDs (as strings) and values are summary strings.
Return ONLY valid JSON, no markdown."""

    raw = call_gemini(prompt, api_key)
    if raw:
        try:
            parsed = json.loads(raw)
            return {int(k): v for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback
    return {cid: f"Community of {len(members)} entities: {', '.join(members[:5])}"
            for cid, members in community_map.items()}


# ============================================================
# STEP 5: Search Engine (Local + Global)
# ============================================================

def local_search(query, G, community_summaries, chunks, api_key):
    """Local search: find relevant entities and their neighborhood."""
    from google import genai
    client = genai.Client(api_key=api_key)

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

    context_parts = []
    for entity in top:
        desc = G.nodes[entity].get("description", "")
        neighbors = list(G.neighbors(entity))
        edge_info = []
        for n in neighbors[:5]:
            if G.has_edge(entity, n):
                descs = G[entity][n].get("descriptions", ["related"])
                edge_info.append(f"  -> {n}: {descs[0]}")
        context_parts.append(f"Entity: {entity}\nDesc: {desc}\nConnections:\n" + "\n".join(edge_info))

    relevant_chunks = [c[:500] for c in chunks if any(e.lower() in c.lower() for e in top[:3])]

    prompt = f"""Answer using ONLY the knowledge graph context and text below.

KNOWLEDGE GRAPH:
{chr(10).join(context_parts[:5])}

TEXT:
{chr(10).join(relevant_chunks[:2])}

QUESTION: {query}

Provide a detailed answer citing specific entities and relationships."""

    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    return response.text


def global_search(query, community_summaries, api_key):
    """Global search: synthesize across community summaries."""
    from google import genai
    client = genai.Client(api_key=api_key)

    summaries_text = "\n\n".join([f"Community {cid}: {s}" for cid, s in community_summaries.items()])

    prompt = f"""Answer using community summaries from a knowledge graph.

COMMUNITY SUMMARIES:
{summaries_text}

QUESTION: {query}

Synthesize across all relevant communities. Highlight themes and connections."""

    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    return response.text


# ============================================================
# STEP 6: Main Pipeline
# ============================================================

def run_pipeline(input_dir, output_dir, api_key):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(" DS Knowledge Navigator - Mini GraphRAG Pipeline")
    print("=" * 60)

    # Step 1
    print("\n[Step 1] Loading documents...")
    docs = load_documents(input_dir)
    print(f"  Loaded {len(docs)} documents")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc["text"], chunk_size=500, overlap=50))
    print(f"  Created {len(all_chunks)} text chunks")

    # Step 2
    print("\n[Step 2] Extracting entities & relationships...")
    all_extractions = []
    for i, chunk in enumerate(all_chunks):
        print(f"  Chunk {i+1}/{len(all_chunks)}...", end=" ")
        extraction = extract_entities_relationships(chunk, api_key)
        all_extractions.append(extraction)
        n_ent = len(extraction.get("entities", []))
        n_rel = len(extraction.get("relationships", []))
        print(f"OK ({n_ent} entities, {n_rel} relationships)")
        time.sleep(2)  # gentle rate limiting

    with open(os.path.join(output_dir, "extractions.json"), "w", encoding="utf-8") as f:
        json.dump(all_extractions, f, indent=2)

    # Step 3
    print("\n[Step 3] Building knowledge graph...")
    G = build_knowledge_graph(all_extractions)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    nx.write_gml(G, os.path.join(output_dir, "knowledge_graph.gml"))

    # Step 4
    print("\n[Step 4] Detecting communities...")
    community_map = detect_communities(G)
    print(f"  Found {len(community_map)} communities")
    for cid, members in community_map.items():
        print(f"    Community {cid}: {len(members)} members - {', '.join(members[:4])}...")

    # Step 5
    print("\n[Step 5] Generating community summaries...")
    summaries = generate_community_summaries(community_map, G, api_key)
    for cid, summary in summaries.items():
        print(f"  Community {cid}: {summary[:100]}...")

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
    print(" Pipeline complete!")
    print(f"  {stats['num_entities']} entities, {stats['num_relationships']} relationships")
    print(f"  {stats['num_communities']} communities detected")
    print(f"  Results saved to: {output_dir}/")
    print("=" * 60)

    return G, community_map, summaries, all_chunks


def interactive_search(G, community_summaries, chunks, api_key):
    print("\n--- Interactive Search (type 'quit' to exit) ---")
    print("  Use 'local <query>' or 'global <query>'\n")

    while True:
        user_input = input("Query > ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.lower().startswith("global "):
            query = user_input[7:]
            print("\n[Global Search]...\n")
            result = global_search(query, community_summaries, api_key)
        else:
            query = user_input.replace("local ", "", 1)
            print("\n[Local Search]...\n")
            result = local_search(query, G, community_summaries, chunks, api_key)

        print(result)
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("Error: Set GEMINI_API_KEY in .env file")
        exit(1)

    G, communities, summaries, chunks = run_pipeline("input", "output", API_KEY)
    interactive_search(G, summaries, chunks, API_KEY)
