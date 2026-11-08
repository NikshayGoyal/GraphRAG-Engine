"""Atomic snapshots and process-local session directories, never shared upload paths."""
import json
import os
import tempfile
from pathlib import Path
import networkx as nx


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.write-', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_index(path, index):
    payload = dict(index)
    payload['graph'] = nx.node_link_data(index['graph'], edges='edges')
    atomic_json(path, payload)


def load_index(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 2:
        raise ValueError('Legacy index: reindex source documents to recover provenance.')
    payload['graph'] = nx.node_link_graph(payload['graph'], edges='edges')
    payload['communities'] = {int(k): v for k, v in payload['communities'].items()}
    payload['reports'] = {int(k): v for k, v in payload['reports'].items()}
    return payload


def new_workspace():
    # Keep this owner in Streamlit session state. GC cleans it after session expiry;
    # explicit reset calls cleanup immediately. No uploaded filenames become paths.
    return tempfile.TemporaryDirectory(prefix='graphrag-session-')
