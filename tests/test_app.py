from pathlib import Path
from streamlit.testing.v1 import AppTest


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_no_key_demo_comparison_benchmark_and_no_rerun_calls(monkeypatch):
    monkeypatch.delenv('MISTRAL_API_KEY', raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == 'GraphRAG Evidence Lab'
    button(app, 'Run comparison').click().run()
    assert not app.exception
    assert len(app.session_state['runs']) == 2
    assert all(r['answer'] is None for r in app.session_state['runs'])
    app.run()
    assert len(app.session_state['runs']) == 2
    button(app, 'Run synthetic benchmark').click().run()
    assert not app.exception
    assert len(app.session_state['benchmark']['rows']) == 18
    button(app, 'Reset session and load demo').click().run()
    assert not app.exception
    assert app.session_state['runs'] == []


def test_live_answer_review_flow_with_mocked_provider(monkeypatch):
    monkeypatch.setenv('MISTRAL_API_KEY', 'fake-test-key')
    from src.extraction import MistralClient
    import json
    def answer(self, prompt, system=None):
        context = json.loads(prompt.split('\n', 1)[1])
        return {'abstain': False, 'claims': [{'text': 'Synthetic test response.', 'citation_ids': [context['sources'][0]['id']]}]}
    monkeypatch.setattr(MistralClient, 'complete', answer)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    next(c for c in app.checkbox if c.label == 'Generate cited answers with Mistral').check()
    button(app, 'Run comparison').click().run()
    assert not app.exception
    assert len(app.session_state['runs']) == 2
    assert all(r['status'] == 'answered' for r in app.session_state['runs'])
    next(s for s in app.selectbox if s.label == 'Answer judgment').select('Correct')
    next(s for s in app.selectbox if s.label == 'Citation support').select('Supported')
    next(b for b in app.button if b.label == 'Save judgment').click().run()
    assert not app.exception
    assert app.session_state['judgments'][0]['correctness'] == 'Correct'
    assert len(app.session_state['runs']) == 2
