"""Document ingestion with stable source IDs and exact character spans."""
import hashlib
import io
import re
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_DOCUMENT_CHARS = 500_000


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _parse_document(name: str, data: bytes) -> dict:
    if not data or len(data) > MAX_FILE_BYTES:
        raise ValueError('Each document must contain data and be at most 5 MB.')
    name = Path(name.replace('\\', '/')).name
    ext = Path(name).suffix.lower()
    if ext == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > 300:
            raise ValueError('PDF exceeds the 300-page limit.')
        pages = [{'page': i + 1, 'text': page.extract_text() or ''}
                 for i, page in enumerate(reader.pages)]
    elif ext == '.docx':
        from docx import Document
        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs]
        paragraphs.extend(' | '.join(c.text for c in row.cells)
                          for table in doc.tables for row in table.rows)
        pages = [{'page': None, 'text': '\n'.join(paragraphs)}]
    elif ext in {'.txt', '.md'}:
        pages = [{'page': None, 'text': data.decode('utf-8-sig')}]
    else:
        raise ValueError('Supported files: TXT, Markdown, PDF and DOCX.')
    if not any(p['text'].strip() for p in pages):
        raise ValueError('No extractable text; scanned PDFs require OCR before upload.')
    if sum(len(p['text']) for p in pages) > MAX_DOCUMENT_CHARS:
        raise ValueError('Document exceeds 500,000 extracted characters.')
    identity = name + '\n' + '\n'.join(p['text'] for p in pages)
    return {'id': digest(identity), 'source': name, 'pages': pages}


def load_documents(input_dir: str) -> list[dict]:
    return [parse_document(p.name, p.read_bytes()) for p in sorted(Path(input_dir).iterdir())
            if p.is_file() and p.suffix.lower() in {'.txt', '.md', '.pdf', '.docx'}]


def chunk_documents(documents: list[dict], chunk_size=220, overlap=35) -> list[dict]:
    if not 0 <= overlap < chunk_size:
        raise ValueError('Require 0 <= overlap < chunk_size.')
    chunks, seen = [], set()
    for doc in documents:
        for page in doc['pages']:
            text = page['text']
            words = list(re.finditer(r'\S+', text))
            for i in range(0, len(words), chunk_size - overlap):
                end = min(i + chunk_size, len(words))
                start_char, end_char = words[i].start(), words[end - 1].end()
                cid = digest(f"{doc['id']}:{page['page']}:{start_char}:{end_char}")
                if cid not in seen:
                    chunks.append({'id': cid, 'document_id': doc['id'], 'source': doc['source'],
                                   'page': page['page'], 'start_char': start_char, 'end_char': end_char,
                                   'text': text[start_char:end_char]})
                    seen.add(cid)
                if end == len(words):
                    break
    return chunks


def chunk_text(text: str, chunk_size=500, overlap=50) -> list[str]:
    doc = {'id': digest(text), 'source': 'text', 'pages': [{'page': None, 'text': text}]}
    return [c['text'] for c in chunk_documents([doc], chunk_size, overlap)]


def parse_document(name: str, data: bytes) -> dict:
    try:
        return _parse_document(name, data)
    except ValueError:
        raise
    except Exception:
        raise ValueError('Could not parse this document. Check that it is a valid, unencrypted supported file.') from None
