import json
import threading
from pathlib import Path

import httpx
import networkx as nx
import pytest

from pipeline import build_index
from src.chunking import chunk_documents, chunk_text, parse_document
from src.community import detect_communities
from src.demo_data import demo_index, demo_questions
from src.evaluation import evaluate
from src.extraction import MistralClient, ProviderError, validate_extraction
from src.graph_builder import build_knowledge_graph
from src.search import run_search, retrieve, validate_answer
from src.storage import load_index, save_index, new_workspace


def extraction():
    return {'entities': [{'name': 'A', 'type': 'organization', 'description': 'A'},
                         {'name': 'B', 'type': 'organization', 'description': 'B'}],
            'relationships': [{'source': 'A', 'target': 'B', 'predicate': 'owns', 'description': 'A owns B', 'weight': 1}]}


class FakeClient:
    model = 'test-model-v1'
    def __init__(self, fail=False):
        self.events, self.calls, self.fail = [], 0, fail
    def complete(self, prompt):
        self.calls += 1
        if self.fail:
            raise ProviderError('network', 'Simulated outage')
        return extraction()


def test_chunk_spans_and_validation():
    text = 'Alpha\n\n beta  gamma\tdelta epsilon'
    doc = parse_document('../../source.txt', text.encode())
    chunks = chunk_documents([doc], chunk_size=3, overlap=1)
    assert doc['source'] == 'source.txt'
    assert len(chunks) == 2
    assert all(text[c['start_char']:c['end_char']] == c['text'] for c in chunks)
    assert chunks == chunk_documents([doc], 3, 1)
    with pytest.raises(ValueError):
        chunk_text(text, 3, 3)
    with pytest.raises(ValueError):
        parse_document('empty.txt', b'   ')


def test_pdf_page_provenance():
    doc = {'id': 'doc', 'source': 'report.pdf', 'pages': [{'page': 1, 'text': 'Page one evidence.'}, {'page': 2, 'text': 'Second page evidence.'}]}
    chunks = chunk_documents([doc])
    assert [c['page'] for c in chunks] == [1, 2]
    assert len({c['id'] for c in chunks}) == 2


def test_direction_predicates_dedup_and_explicit_aliases():
    first = {**extraction(), 'chunk_id': 'c1'}
    second = {**extraction(), 'chunk_id': 'c2'}
    second['relationships'] = [{'source': 'B', 'target': 'A', 'predicate': 'supplies', 'description': 'B supplies A'}]
    graph = build_knowledge_graph([first, first, second], aliases={'B': 'Boreal'})
    assert graph.has_edge('A', 'BOREAL', 'owns')
    assert graph.has_edge('BOREAL', 'A', 'supplies')
    assert graph['A']['BOREAL']['owns']['weight'] == 1
    assert graph['A']['BOREAL']['owns']['chunk_ids'] == ['c1']


def test_roundtrip_community_and_provenance(tmp_path):
    original = demo_index()
    save_index(tmp_path / 'index.json', original)
    restored = load_index(tmp_path / 'index.json')
    assert nx.utils.graphs_equal(original['graph'], restored['graph'])
    assert all('community' in attrs for _, attrs in restored['graph'].nodes(data=True))
    assert restored['chunks'] == original['chunks']
    assert restored['reports'] == original['reports']


def test_empty_and_edgeless_communities():
    assert detect_communities(nx.MultiDiGraph()) == {}
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(['A', 'B'])
    assert detect_communities(graph) == {0: ['A'], 1: ['B']}


@pytest.mark.parametrize('bad', [None, {}, {'entities': [], 'relationships': [{}]},
    {'entities': [{'name': 'A'}], 'relationships': []}])
def test_malformed_extractions_rejected(bad):
    with pytest.raises(ValueError):
        validate_extraction(bad)


def test_invalid_endpoint_and_weight():
    data = extraction()
    data['relationships'][0]['target'] = 'UNKNOWN'
    with pytest.raises(ValueError):
        validate_extraction(data)
    data = extraction()
    data['relationships'][0]['weight'] = float('nan')
    with pytest.raises(ValueError):
        validate_extraction(data)


def test_incremental_cache_and_failure_preserves_index(tmp_path):
    docs = [parse_document('a.txt', b'A owns B.')]
    client = FakeClient()
    first = build_index(docs, tmp_path, client)
    assert first['status'] == 'complete' and client.calls == 1
    second = build_index(docs, tmp_path, client)
    assert second['stats']['reused_chunks'] == 1 and client.calls == 1
    failed = build_index(docs + [parse_document('b.txt', b'New evidence.')], tmp_path, FakeClient(fail=True))
    assert failed['status'] == 'partial'
    # Second successful run changed timings, so compare current authoritative index identity.
    assert load_index(tmp_path / 'index.json')['index_id'] == first['index_id']
    assert len(failed['failures']) == 1
    assert load_index(tmp_path / 'index.json')['stats']['chunks'] == 1
    resumed = build_index(docs + [parse_document('b.txt', b'New evidence.')], tmp_path, client)
    assert resumed['status'] == 'complete' and resumed['stats']['reused_chunks'] == 1
    assert client.calls == 2


def test_corrupt_cache_rebuilt(tmp_path):
    docs = [parse_document('a.txt', b'A owns B.')]
    client = FakeClient()
    build_index(docs, tmp_path, client)
    next((tmp_path / 'cache').glob('*.json')).write_text('broken')
    assert build_index(docs, tmp_path, client)['status'] == 'complete'
    assert client.calls == 2


def test_independent_workspace_builds():
    with new_workspace() as a, new_workspace() as b:
        assert a != b
        def run(path, name):
            build_index([parse_document(name, name.encode())], path, FakeClient())
        threads = [threading.Thread(target=run, args=(a, 'first.txt')), threading.Thread(target=run, args=(b, 'second.txt'))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert load_index(Path(a) / 'index.json')['chunks'][0]['source'] == 'first.txt'
        assert load_index(Path(b) / 'index.json')['chunks'][0]['source'] == 'second.txt'


def test_unknown_citations_fail_closed():
    evidence = [{'id': 'known'}]
    with pytest.raises(ValueError):
        validate_answer({'abstain': False, 'claims': [{'text': 'Claim', 'citation_ids': ['invented']}]}, evidence)
    with pytest.raises(ValueError):
        validate_answer({'abstain': False, 'claims': [{'text': 'Claim', 'citation_ids': []}]}, evidence)
    assert validate_answer({'abstain': True, 'claims': []}, evidence)['abstain']


def test_graph_recovers_linked_evidence_and_budget():
    index = demo_index()
    query = demo_questions()[2]['question']
    graph = retrieve(query, index, 'Graph + text', top_k=3)
    assert '03_cedar_risk.txt' in {c['source'] for c in graph['evidence']}
    assert graph['edges']
    bounded = retrieve(query, index, 'Graph + text', top_k=2, max_chars=200)
    assert len(bounded['evidence']) <= 2
    assert sum(len(c['text']) for c in bounded['evidence']) <= 200


def test_retrieval_only_not_answer_accuracy():
    result = run_search('Aster Holdings', demo_index(), 'TF-IDF baseline')
    assert result['generation'] == 'retrieval_only' and result['answer'] is None
    report = evaluate(demo_index(), demo_questions())
    assert len(report['rows']) == 18
    assert all(r['required_source_recall'] is None for r in report['rows'] if r['unanswerable'])
    assert all(row['scored_questions'] == 5 for row in report['summary'])


def test_provider_auth_no_retry_no_secret_leak():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(401, text='sensitive error content')
    client = MistralClient('secret-key', transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError, match='HTTP 401') as error:
        client.complete('hello')
    assert error.value.code == 'authentication'
    assert 'secret' not in str(error.value) and 'sensitive' not in str(error.value)
    assert len(calls) == 1


def test_retry_usage_and_timeout():
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        if count < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"ok":true}'}}], 'usage': {'prompt_tokens': 12}, 'model': 'snapshot'})
    client = MistralClient('key', transport=httpx.MockTransport(handler), sleep=lambda _: None)
    assert client.complete('hello') == {'ok': True}
    assert client.events[0]['attempts'] == 3
    assert client.events[0]['usage']['prompt_tokens'] == 12
    assert client.events[0]['model'] == 'snapshot'


def test_provider_error_preserves_retrieved_evidence():
    result = run_search('Aster Holdings', demo_index(), 'Graph + text', FakeClient(fail=True))
    assert result['status'] == 'error' and result['error_code'] == 'network'
    assert result['evidence'] and result['answer'] is None


@pytest.mark.parametrize('name', ['broken.pdf', 'broken.docx'])
def test_corrupt_documents_have_actionable_error(name):
    with pytest.raises(ValueError):
        parse_document(name, b'This is not a valid binary document')


def test_valid_claims_and_absent_evidence():
    index = demo_index()
    class AnswerClient(FakeClient):
        def complete(self, prompt):
            self.calls += 1
            # Use a known source actually selected by this retrieval.
            cid = retrieve('Who owns Boreal Components?', index)['evidence'][0]['id']
            return {'abstain': False, 'claims': [{'text': 'Aster Holdings owns Boreal Components.', 'citation_ids': [cid]}]}
    result = run_search('Who owns Boreal Components?', index, 'Graph + text', AnswerClient())
    assert result['status'] == 'answered' and not result['answer']['abstain']
    client = AnswerClient()
    result = run_search('xyzzyquux', index, 'TF-IDF baseline', client)
    assert result['answer']['abstain'] and client.calls == 0


def test_provider_timeout_is_bounded():
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout('timeout', request=request)
    client = MistralClient('key', transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderError) as error:
        client.complete('hello')
    assert error.value.code == 'network' and count == 3


def test_changed_model_invalidates_extraction_cache(tmp_path):
    docs = [parse_document('a.txt', b'A owns B.')]
    client = FakeClient()
    build_index(docs, tmp_path, client)
    client.model = 'different-snapshot'
    result = build_index(docs, tmp_path, client)
    assert result['stats']['reused_chunks'] == 0 and client.calls == 2
