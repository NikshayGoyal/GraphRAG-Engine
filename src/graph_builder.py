"""Directed evidence graph; alias merges require an explicit reviewed mapping."""
import re
import unicodedata
import networkx as nx


def canonical(name):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', name)).strip().upper()


def build_knowledge_graph(extractions: list[dict], aliases=None) -> nx.MultiDiGraph:
    aliases = {canonical(k): canonical(v) for k, v in (aliases or {}).items()}
    def key(name):
        name = canonical(name)
        return aliases.get(name, name)
    graph = nx.MultiDiGraph()
    for item in extractions:
        cid = item['chunk_id']
        for entity in item['entities']:
            name = key(entity['name'])
            if name not in graph:
                graph.add_node(name, type=entity['type'], description=entity['description'], chunk_ids=[], aliases=[])
            attrs = graph.nodes[name]
            if cid not in attrs['chunk_ids']:
                attrs['chunk_ids'].append(cid)
            if entity['name'] not in attrs['aliases']:
                attrs['aliases'].append(entity['name'])
        for rel in item['relationships']:
            source, target = key(rel['source']), key(rel['target'])
            predicate = rel['predicate'].strip().lower()
            if graph.has_edge(source, target, predicate):
                attrs = graph[source][target][predicate]
                if cid not in attrs['chunk_ids']:
                    attrs['chunk_ids'].append(cid)
                    attrs['weight'] += 1
                if rel['description'] not in attrs['descriptions']:
                    attrs['descriptions'].append(rel['description'])
            else:
                graph.add_edge(source, target, key=predicate, predicate=predicate, weight=1,
                               descriptions=[rel['description']], chunk_ids=[cid])
    return graph


def community_projection(graph):
    projected = nx.Graph()
    projected.add_nodes_from(graph.nodes)
    for source, target, attrs in graph.edges(data=True):
        if source == target:
            continue
        weight = projected.get_edge_data(source, target, {}).get('weight', 0) + attrs['weight']
        projected.add_edge(source, target, weight=weight)
    return projected
