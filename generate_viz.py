"""Generate the interactive HTML visualization from the knowledge graph."""
import json
import networkx as nx
import os

def generate_visualization():
    # Load graph
    G = nx.read_gml(os.path.join("output", "knowledge_graph.gml"))
    stats = json.load(open(os.path.join("output", "graph_stats.json"), encoding="utf-8"))
    
    # Build JSON data for the visualization
    node_list = []
    id_map = {}
    for i, (node, data) in enumerate(G.nodes(data=True)):
        id_map[node] = i
        comm = data.get("community", 0)
        degree = G.degree(node)
        node_list.append({
            "id": i, "label": node,
            "type": data.get("type", "unknown"),
            "description": data.get("description", ""),
            "community": comm,
            "degree": degree
        })
    
    edge_list = []
    for src, tgt, data in G.edges(data=True):
        if src in id_map and tgt in id_map:
            edge_list.append({
                "source": id_map[src], "target": id_map[tgt],
                "weight": data.get("weight", 1),
                "descriptions": data.get("descriptions", ["related"])
            })
    
    graph_json = json.dumps({"nodes": node_list, "edges": edge_list})
    
    # Read template and inject data
    with open("visualization_template.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    html = html.replace("GRAPH_DATA_PLACEHOLDER", graph_json)
    
    with open("knowledge_graph.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Visualization generated: knowledge_graph.html")
    print(f"  {len(node_list)} nodes, {len(edge_list)} edges")
    print("  Open in browser to explore!")

if __name__ == "__main__":
    generate_visualization()
