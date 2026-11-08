"""GraphRAG evidence lab: isolated uploads, comparable retrieval and auditable sources."""
import html
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pyvis.network import Network

from pipeline import build_index
from src.chunking import parse_document
from src.demo_data import demo_index, demo_questions
from src.evaluation import evaluate, judgment_summary
from src.extraction import MistralClient
from src.search import MODES, run_search
from src.storage import atomic_json, new_workspace

load_dotenv()
st.set_page_config(page_title='GraphRAG Evidence Lab', page_icon='🔎', layout='wide')

if 'workspace_owner' not in st.session_state:
    st.session_state.workspace_owner = new_workspace()
    st.session_state.index = demo_index()
    st.session_state.dataset = 'Synthetic demo'
    st.session_state.runs = []
    st.session_state.judgments = []
    st.session_state.latest = []
    st.session_state.attempt = None
    st.session_state.question = demo_questions()[2]['question']

workspace = Path(st.session_state.workspace_owner.name)
api_key = os.getenv('MISTRAL_API_KEY', '')
if not api_key:
    try:
        api_key = st.secrets.get('MISTRAL_API_KEY', '')
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        api_key = ''


def clear_results():
    st.session_state.latest = []
    st.session_state.attempt = None
    st.session_state.pop('benchmark', None)


def reset_workspace():
    st.session_state.workspace_owner.cleanup()
    st.session_state.workspace_owner = new_workspace()
    st.session_state.index = demo_index()
    st.session_state.dataset = 'Synthetic demo'
    st.session_state.runs = []
    st.session_state.judgments = []
    clear_results()
    st.rerun()


def usage_of(result):
    usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    for event in result.get('provider_events', []):
        for key in usage:
            usage[key] += event.get('usage', {}).get(key, 0)
    return usage


with st.sidebar:
    st.header('Workspace')
    st.caption('Uploads and results are isolated to this browser session. Export results before leaving; session storage is temporary.')
    st.info('Synthetic demo: hand-authored documents and graph. Retrieval is real; no prerecorded answers or invented scores.')
    with st.form('upload_form'):
        files = st.file_uploader('Documents', type=['txt', 'md', 'pdf', 'docx'], accept_multiple_files=True)
        st.caption('Up to 20 files, 5 MB each; at most 200 chunks. PDF citations retain page numbers. DOCX/text citations use character spans.')
        acknowledged = st.checkbox('Process these documents with Mistral')
        build = st.form_submit_button('Build document index', disabled=not bool(api_key))
    if not api_key:
        st.caption('Live indexing and answers require a server-side MISTRAL_API_KEY. No key is needed for the demo or evaluation.')
    if build:
        if not acknowledged or not files:
            st.warning('Choose documents and enable processing with Mistral.')
        elif len(files) > 20:
            st.error('Please upload at most 20 files.')
        else:
            try:
                names = [Path(f.name.replace('\\', '/')).name for f in files]
                if len(set(names)) != len(names):
                    raise ValueError('Document filenames must be unique within a collection.')
                docs = [parse_document(f.name, f.getvalue()) for f in files]
                progress = st.progress(0.0)
                attempt = build_index(docs, workspace, MistralClient(api_key), progress=progress.progress)
                st.session_state.attempt = attempt
                if attempt['status'] == 'complete':
                    st.session_state.index = attempt
                    st.session_state.dataset = 'Uploaded documents'
                    st.session_state.latest = []
                    st.session_state.pop('benchmark', None)
                    st.success(f"Index ready. Reused {attempt['stats']['reused_chunks']} cached chunks.")
                else:
                    st.error('Build incomplete. The previous index remains active. Retry to reuse successful chunks.')
            except (ValueError, OSError) as error:
                st.error(str(error))
    if st.button('Reset session and load demo'):
        reset_workspace()
    st.caption('NetworkX · TF-IDF · Mistral · Streamlit')

index = st.session_state.index
st.title('GraphRAG Evidence Lab')
st.write('Compare retrieval. Inspect the source. Record what works.')
st.caption(f"{st.session_state.dataset} · Index {index['index_id'][:10]} · {index['model']}")
cols = st.columns(4)
for col, key in zip(cols, ['documents', 'entities', 'relationships', 'communities']):
    col.metric(key.title(), index['stats'][key])

compare_tab, graph_tab, eval_tab, health_tab = st.tabs(['Compare & inspect', 'Knowledge graph', 'Evaluation', 'Index health'])

with compare_tab:
    if st.session_state.dataset == 'Synthetic demo':
        sample = st.selectbox('Example question', ['Custom question'] + [q['question'] for q in demo_questions()])
        if st.button('Use example', disabled=sample == 'Custom question'):
            st.session_state.question = sample
    with st.form('search_form'):
        question = st.text_input('Question', key='question', max_chars=2000)
        modes = st.multiselect('Retrieval methods', MODES, default=list(MODES[:2]))
        a, b = st.columns(2)
        top_k = a.slider('Maximum passages per method', 1, 10, 3)
        generate = b.checkbox('Generate cited answers with Mistral', disabled=not bool(api_key))
        submitted = st.form_submit_button('Run comparison', type='primary')
    st.caption('All methods use the same passage limit and 12,000-character source budget. TF-IDF is a lexical baseline, not dense-vector RAG. Reports are extractive community records.')
    if submitted:
        if not modes or not question.strip():
            st.warning('Enter a question and select at least one method.')
        else:
            latest = []
            with st.spinner('Retrieving evidence' + (' and generating answers…' if generate else '…')):
                for mode in modes:
                    result = run_search(question, index, mode, MistralClient(api_key) if generate else None, top_k)
                    result['run_id'] = uuid.uuid4().hex
                    latest.append(result)
            st.session_state.latest = latest
            st.session_state.runs = (st.session_state.runs + latest)[-100:]
            atomic_json(workspace / 'runs.json', st.session_state.runs)
    latest = st.session_state.latest
    if latest:
        for column, result in zip(st.columns(len(latest)), latest):
            with column:
                st.subheader(result['mode'])
                st.caption(f"{result['seconds']:.3f}s total · {result['retrieval_seconds']:.3f}s retrieval · {result['context_chars']:,} source characters")
                if result['status'] == 'error':
                    st.error(f"{result['error_code']}: {result['error']}")
                    st.caption('Retrieved passages remain available below; no answer was fabricated.')
                elif result['answer']:
                    answer = result['answer']
                    if answer['abstain']:
                        st.warning(answer['reason'])
                    else:
                        st.caption('Citation IDs validated against retrieved passages. Factual support still requires review.')
                        for claim in answer['claims']:
                            st.write(claim['text'])
                            st.caption('Sources: ' + ', '.join(cid[:8] for cid in claim['citation_ids']))
                else:
                    st.info('Retrieval only — no generated answer or answer-quality score.')
                if result['provider_events']:
                    usage = usage_of(result)
                    st.caption(f"Tokens: {usage['prompt_tokens']} input / {usage['completion_tokens']} output")
                    try:
                        input_price = float(os.environ['MISTRAL_INPUT_USD_PER_MILLION'])
                        output_price = float(os.environ['MISTRAL_OUTPUT_USD_PER_MILLION'])
                        if input_price >= 0 and output_price >= 0:
                            price = (usage['prompt_tokens'] * input_price + usage['completion_tokens'] * output_price) / 1e6
                            st.caption(f'Estimated successful-call cost: ${price:.6f} (configured prices; failed requests excluded)')
                    except (KeyError, ValueError):
                        st.caption('Dollar cost not calculated: configure actual provider prices.')
                st.markdown('**Source passages**')
                for passage in result['evidence']:
                    location = f"page {passage['page']}" if passage['page'] else f"characters {passage['start_char']}–{passage['end_char']}"
                    with st.expander(f"{passage['id'][:8]} · {passage['source']} · {location}"):
                        st.code(passage['text'], language=None, wrap_lines=True)
                        st.caption('Selection: ' + '; '.join(passage['reasons']))
                if not result['evidence']:
                    st.warning('No relevant passages found.')
                if result['edges']:
                    with st.expander('Retrieved graph relationships'):
                        st.dataframe(pd.DataFrame(result['edges']), hide_index=True)
                if result['communities']:
                    st.caption('Communities consulted: ' + ', '.join(map(str, result['communities'])))
                if result['answer']:
                    with st.form(f"review_{result['run_id']}"):
                        correctness = st.selectbox('Answer judgment', ['Unreviewed', 'Correct', 'Partially correct', 'Incorrect', 'Appropriate abstention', 'Inappropriate abstention'])
                        support = st.selectbox('Citation support', ['Unreviewed', 'Supported', 'Partially supported', 'Unsupported', 'Not applicable'])
                        note = st.text_area('Reviewer notes')
                        save = st.form_submit_button('Save judgment')
                    if save:
                        judgment = {'run_id': result['run_id'], 'index_id': result['index_id'], 'mode': result['mode'],
                                    'query': result['query'], 'correctness': correctness, 'citation_support': support, 'notes': note}
                        st.session_state.judgments = [j for j in st.session_state.judgments if j['run_id'] != result['run_id']] + [judgment]
                        atomic_json(workspace / 'judgments.json', st.session_state.judgments)
                        st.success('Judgment saved for this session.')

with graph_tab:
    st.caption('Directed relationships retain their source evidence. Community detection uses a weighted undirected projection.')
    selected = st.selectbox('Community', ['All'] + list(index['communities']))
    graph = index['graph']
    names = sorted(n for n, attrs in graph.nodes(data=True) if selected == 'All' or attrs['community'] == selected)
    st.caption(f'Showing up to 150 of {len(names)} matching entities.')
    names = names[:150]
    ids = {name: i for i, name in enumerate(names)}
    net = Network(height='480px', width='100%', directed=True, bgcolor='#f8fafc', font_color='#172554', cdn_resources='in_line')
    for name in names:
        attrs = graph.nodes[name]
        net.add_node(ids[name], label=html.escape(name), title=html.escape(attrs['description']), group=attrs['community'])
    for source, target, attrs in graph.edges(data=True):
        if source in ids and target in ids:
            net.add_edge(ids[source], ids[target], title=html.escape(attrs['predicate']), value=attrs['weight'])
    net.set_options('{"physics":{"stabilization":{"iterations":100}},"interaction":{"hover":true}}')
    st.components.v1.html(net.generate_html(), height=500)
    if names:
        entity = st.selectbox('Inspect entity evidence', names)
        source_ids = set(graph.nodes[entity]['chunk_ids'])
        for chunk in index['chunks']:
            if chunk['id'] in source_ids:
                with st.expander(f"{chunk['source']} · {chunk['id'][:8]}"):
                    st.code(chunk['text'], language=None, wrap_lines=True)

with eval_tab:
    st.subheader('Reproducible retrieval check')
    st.write('Six synthetic questions test the wiring of retrieval and graph expansion. This is a development fixture, not an independent benchmark or a claim about industry performance.')
    if st.button('Run synthetic benchmark'):
        st.session_state.benchmark = evaluate(demo_index(), demo_questions(), top_k=3)
    if 'benchmark' in st.session_state:
        benchmark = st.session_state.benchmark
        st.caption('Always uses the bundled synthetic corpus, independent of uploaded documents. Measures required-source recall at 3 passages; unanswerable questions are excluded from recall aggregates.')
        st.dataframe(pd.DataFrame(benchmark['summary']), hide_index=True)
        with st.expander('Question-level results and failure cases'):
            st.dataframe(pd.DataFrame(benchmark['rows']), hide_index=True)
        st.download_button('Download benchmark JSON', json.dumps(benchmark, indent=2), 'retrieval_benchmark.json', 'application/json')
    st.subheader('Human answer judgments')
    reviewed = [j for j in st.session_state.judgments if j['correctness'] != 'Unreviewed' and j['index_id'] == index['index_id']]
    st.caption('Current index only. Counts reflect your reviews, not an automated truth score; multiple runs of one question can be counted.')
    if reviewed:
        st.dataframe(pd.DataFrame(judgment_summary(reviewed)), hide_index=True)
    else:
        st.info('No reviewed generated answers yet.')
    if st.session_state.runs:
        export = {'runs': st.session_state.runs, 'judgments': st.session_state.judgments}
        st.download_button('Export session runs and judgments', json.dumps(export, indent=2), 'graphrag_evaluation.json', 'application/json')
        st.dataframe(pd.DataFrame([{'mode': r['mode'], 'query': r['query'], 'status': r['status'], 'seconds': r['seconds'],
                                   'generation': r['generation']} for r in st.session_state.runs]), hide_index=True)

with health_tab:
    st.json({'status': index['status'], 'stats': index['stats'], 'prompt_version': index['prompt_version']})
    st.caption('A complete index means all chunks were processed, not that all extracted facts are correct. Cached extraction is reused; graph and communities are rebuilt from the current document set.')
    if st.session_state.attempt:
        attempt = st.session_state.attempt
        st.write('Latest build attempt:', attempt['status'])
        st.json(attempt['stats'])
        if attempt['failures']:
            st.dataframe(pd.DataFrame(attempt['failures']), hide_index=True)
    st.write('Known limits: no OCR, dense-vector baseline, automatic alias disambiguation, authentication, durable multi-user storage, or independent accuracy evaluation yet.')
