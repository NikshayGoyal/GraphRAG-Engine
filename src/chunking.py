"""Document loading and text chunking utilities."""

from pathlib import Path


def load_documents(input_dir: str) -> list[dict]:
    """Load all text files from the input directory.

    Args:
        input_dir: Path to directory containing .txt files.

    Returns:
        List of dicts with 'id', 'text', and 'source' keys.
    """
    docs = []
    for f in sorted(Path(input_dir).glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        docs.append({"id": f.stem, "text": text, "source": f.name})
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: Input text to split.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks
