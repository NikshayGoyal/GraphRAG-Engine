"""Retrieval metrics only; unreviewed answer quality is never manufactured."""
import math
import statistics
from .search import MODES, run_search


def evaluate(index, questions, top_k=3):
    rows = []
    valid_sources = {c['source'] for c in index['chunks']}
    for question in questions:
        expected = set(question['required_sources'])
        if not expected <= valid_sources:
            raise ValueError('Benchmark references source files absent from this index.')
        for mode in MODES:
            result = run_search(question['question'], index, mode, top_k=top_k)
            found = {c['source'] for c in result['evidence']}
            rows.append({'question_id': question['id'], 'category': question['category'], 'mode': mode,
                         'required_source_recall': len(found & expected) / len(expected) if expected else None,
                         'all_required_sources': expected <= found if expected else None,
                         'retrieval_seconds': result['seconds'], 'retrieved_sources': sorted(found),
                         'unanswerable': question.get('unanswerable', False)})
    summary = []
    for mode in MODES:
        selected = [r for r in rows if r['mode'] == mode]
        scored = [r for r in selected if r['required_source_recall'] is not None]
        timings = sorted(r['retrieval_seconds'] for r in selected)
        summary.append({'mode': mode, 'scored_questions': len(scored),
                        'mean_required_source_recall': statistics.mean(r['required_source_recall'] for r in scored) if scored else None,
                        'all_sources_rate': statistics.mean(r['all_required_sources'] for r in scored) if scored else None,
                        'p50_seconds': statistics.median(timings) if timings else None,
                        'p95_seconds': timings[min(len(timings)-1, math.ceil(.95*len(timings))-1)] if timings else None})
    return {'dataset': 'synthetic-fixture-v1', 'scope': 'retrieval-only; not answer accuracy or external validation',
            'index_id': index['index_id'], 'top_k': top_k, 'summary': summary, 'rows': rows}


def judgment_summary(rows):
    result = []
    for mode in sorted({r['mode'] for r in rows}):
        subset = [r for r in rows if r['mode'] == mode]
        result.append({'mode': mode, 'reviewed_runs': len(subset),
                       'correct': sum(r.get('correctness') == 'Correct' for r in subset),
                       'incorrect': sum(r.get('correctness') == 'Incorrect' for r in subset),
                       'supported_citations': sum(r.get('citation_support') == 'Supported' for r in subset)})
    return result
