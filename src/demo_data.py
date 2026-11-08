"""Hand-authored synthetic fixture. No model-generated results or benchmark claims."""
from .chunking import parse_document, chunk_documents, digest
from .graph_builder import build_knowledge_graph
from .community import detect_communities, generate_community_summaries

ROWS = [
    ('01_aster_report.txt', 'Aster Holdings owns Boreal Components. Boreal Components manufactures battery packs.',
     [('Aster Holdings', 'owns', 'Boreal Components')]),
    ('02_boreal_operations.txt', 'Boreal Components operates the Cedar Plant. Cedar Plant is located in Delta Region.',
     [('Boreal Components', 'operates', 'Cedar Plant'), ('Cedar Plant', 'located_in', 'Delta Region')]),
    ('03_cedar_risk.txt', 'Cedar Plant depends on water from Echo Reservoir. Drought at Echo Reservoir can interrupt battery production at Cedar Plant.',
     [('Cedar Plant', 'depends_on', 'Echo Reservoir'), ('Echo Reservoir', 'exposes_to', 'Drought')]),
    ('04_falcon_report.txt', 'Falcon Logistics supplies transport services to Boreal Components. Falcon Logistics reports fuel-price volatility as a material operating risk.',
     [('Falcon Logistics', 'supplies', 'Boreal Components'), ('Falcon Logistics', 'exposes_to', 'Fuel-price volatility')]),
    ('05_grove_report.txt', 'Grove Foods operates in Delta Region. Grove Foods reports drought and fuel-price volatility as operating risks.',
     [('Grove Foods', 'operates_in', 'Delta Region'), ('Grove Foods', 'exposes_to', 'Drought'),
      ('Grove Foods', 'exposes_to', 'Fuel-price volatility')]),
    ('06_harbor_report.txt', 'Harbor Software sells accounting software to independent retailers. Harbor Software reports employee retention as an operating risk.',
     [('Harbor Software', 'sells', 'Accounting software'), ('Harbor Software', 'exposes_to', 'Employee retention')]),
]


def demo_index():
    docs = [parse_document(name, text.encode()) for name, text, _ in ROWS]
    chunks = chunk_documents(docs)
    extractions = []
    for chunk, (_, _, relations) in zip(chunks, ROWS):
        names = sorted({name for source, _, target in relations for name in (source, target)})
        extractions.append({'chunk_id': chunk['id'],
            'entities': [{'name': n, 'type': 'demo_entity', 'description': n} for n in names],
            'relationships': [{'source': s, 'target': t, 'predicate': p, 'description': f'{s} {p} {t}', 'weight': 1}
                              for s, p, t in relations]})
    graph = build_knowledge_graph(extractions)
    communities = detect_communities(graph)
    return {'schema_version': 2, 'index_id': digest(str(ROWS)), 'graph': graph, 'chunks': chunks,
            'documents': docs, 'communities': communities,
            'reports': generate_community_summaries(communities, graph), 'status': 'complete',
            'failures': [], 'model': 'hand-authored synthetic fixture', 'prompt_version': 'fixture-v1',
            'provider_events': [], 'stats': {'documents': len(docs), 'chunks': len(chunks),
            'entities': len(graph), 'relationships': graph.number_of_edges(), 'communities': len(communities)}}


def demo_questions():
    # Required source files are authored explicitly, not inferred from retriever results.
    return [
        {'id': 'fact-1', 'category': 'factual', 'question': 'Who owns Boreal Components?',
         'required_sources': ['01_aster_report.txt']},
        {'id': 'fact-2', 'category': 'factual', 'question': 'Where is Cedar Plant located?',
         'required_sources': ['02_boreal_operations.txt']},
        {'id': 'multi-1', 'category': 'multi_hop', 'question': 'Which reservoir supports the plant operated by the company owned by Aster Holdings?',
         'required_sources': ['01_aster_report.txt', '02_boreal_operations.txt', '03_cedar_risk.txt']},
        {'id': 'multi-2', 'category': 'multi_hop', 'question': 'Which transport supplier serves the company owned by Aster Holdings?',
         'required_sources': ['01_aster_report.txt', '04_falcon_report.txt']},
        {'id': 'theme-1', 'category': 'thematic', 'question': 'Which operating risks recur across companies?',
         'required_sources': ['03_cedar_risk.txt', '04_falcon_report.txt', '05_grove_report.txt']},
        {'id': 'unknown-1', 'category': 'unanswerable', 'question': 'What was the revenue of Aster Holdings in 2025?',
         'required_sources': [], 'unanswerable': True},
    ]
