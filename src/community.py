"""Weighted communities and explicitly extractive, provenance-preserving reports."""
import networkx as nx
from .graph_builder import community_projection


def detect_communities(graph):
    projected = community_projection(graph)
    if not projected:
        return {}
    groups = (nx.community.greedy_modularity_communities(projected, weight='weight')
              if projected.number_of_edges() else [{n} for n in sorted(projected)])
    groups = sorted((sorted(g) for g in groups), key=lambda g: (-len(g), g))
    for cid, members in enumerate(groups):
        for node in members:
            graph.nodes[node]['community'] = cid
    return dict(enumerate(groups))


def generate_community_summaries(community_map, graph, api_key=None):
    reports = {}
    for cid, members in community_map.items():
        evidence = set()
        facts = []
        for name in members:
            evidence.update(graph.nodes[name]['chunk_ids'])
        for source, target, attrs in graph.edges(data=True):
            if source in members and target in members:
                facts.append(f"{source} — {attrs['predicate']} → {target}")
                evidence.update(attrs['chunk_ids'])
        reports[cid] = {'text': '; '.join(facts) or ', '.join(members),
                        'chunk_ids': sorted(evidence), 'kind': 'extractive', 'members': members}
    return reports
