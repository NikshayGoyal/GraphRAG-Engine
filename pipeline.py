"""Incremental extraction cache, explicit failures, and atomic index publication."""
import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from src.chunking import chunk_documents, load_documents, digest
from src.extraction import MistralClient, PROMPT_VERSION, extract, validate_extraction, ProviderError
from src.graph_builder import build_knowledge_graph
from src.community import detect_communities, generate_community_summaries
from src.storage import atomic_json, save_index

MAX_CHUNKS = 200


def build_index(documents, workspace, client, aliases=None, progress=None):
    chunks = chunk_documents(documents)
    if not chunks or len(chunks) > MAX_CHUNKS:
        raise ValueError(f'Provide 1–{MAX_CHUNKS} chunks; reduce the document collection.')
    workspace = Path(workspace)
    cache = workspace / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    started, events_start = time.perf_counter(), len(client.events)
    extractions, failures, reused = [], [], 0
    for i, chunk in enumerate(chunks):
        cache_key = digest(f"{PROMPT_VERSION}:{client.model}:{chunk['text']}")
        path = cache / f'{cache_key}.json'
        try:
            value = None
            if path.exists():
                try:
                    value = validate_extraction(json.loads(path.read_text()))
                    reused += 1
                except (ValueError, TypeError, KeyError):
                    value = None
            if value is None:
                value = extract(chunk['text'], client)
                atomic_json(path, value)
            extractions.append({**value, 'chunk_id': chunk['id']})
        except (ProviderError, ValueError) as error:
            failures.append({'chunk_id': chunk['id'], 'source': chunk['source'],
                             'code': getattr(error, 'code', 'invalid_extraction'), 'message': str(error)})
            if getattr(error, 'code', '') in ('authentication', 'missing_key'):
                # Do not burn repeated requests on a configuration error.
                failures.extend({'chunk_id': c['id'], 'source': c['source'], 'code': 'not_attempted',
                                 'message': 'Indexing stopped after authentication/configuration failure.'}
                                for c in chunks[i + 1:])
                break
        if progress:
            progress((i + 1) / len(chunks))
    graph = build_knowledge_graph(extractions, aliases)
    communities = detect_communities(graph)
    reports = generate_community_summaries(communities, graph)
    result = {'schema_version': 2, 'graph': graph, 'communities': communities, 'reports': reports,
              'chunks': chunks, 'documents': documents, 'failures': failures,
              'status': 'partial' if failures and extractions else 'failed' if failures else 'complete',
              'model': client.model, 'prompt_version': PROMPT_VERSION,
              'stats': {'documents': len(documents), 'chunks': len(chunks), 'reused_chunks': reused,
                        'indexed_chunks': len(extractions), 'entities': len(graph),
                        'relationships': graph.number_of_edges(), 'communities': len(communities),
                        'seconds': time.perf_counter() - started},
              'provider_events': client.events[events_start:]}
    result['index_id'] = digest(json.dumps({'chunks': chunks, 'extractions': extractions,
                                           'model': client.model, 'version': PROMPT_VERSION,
                                           'aliases': aliases or {}}, sort_keys=True))
    # A failed rebuild cannot replace the last usable index. Cached successes remain resumable.
    save_index(workspace / 'last_attempt.json', result)
    if result['status'] == 'complete':
        save_index(workspace / 'index.json', result)
    return result


def run_pipeline(input_dir, output_dir, api_key):
    return build_index(load_documents(input_dir), output_dir, MistralClient(api_key))


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/documents')
    parser.add_argument('--output', default='output')
    args = parser.parse_args()
    result = run_pipeline(args.input, args.output, os.getenv('MISTRAL_API_KEY', ''))
    print(json.dumps({'status': result['status'], 'stats': result['stats'], 'failures': result['failures']}, indent=2))
    raise SystemExit(0 if result['status'] == 'complete' else 1)
