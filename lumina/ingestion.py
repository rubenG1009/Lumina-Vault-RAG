import hashlib
import io
import re
from pathlib import Path
from typing import Any

from lumina.models import Chunk

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 200


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def ingest_bytes(
    data: bytes, source: str, metadata: dict[str, Any] | None = None,
    chunk_size: int = 900, overlap: int = 100,
) -> list[Chunk]:
    """Offsets refer to raw Markdown or extracted page text; never execute document content."""
    if not data or len(data) > MAX_BYTES:
        raise ValueError("File must contain 1 byte to 10 MiB")
    if not 100 <= chunk_size <= 4000 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size/overlap")
    suffix = Path(source).suffix.lower()
    title = Path(source).stem
    sections: list[tuple[str, str, int | None, int]] = []
    if suffix in {".md", ".markdown"}:
        text = data.decode("utf-8")
        heading = ""
        position = 0
        # Headings are metadata; they are not instructions or system messages.
        for match in re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE):
            if match.start() > position:
                sections.append((text[position:match.start()], heading, None, position))
            heading = match.group(1).strip()
            if not sections and not position:
                title = heading
            position = match.end()
        sections.append((text[position:], heading, None, position))
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are unsupported")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError("PDF exceeds 200 pages")
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)[:300]
        for page, obj in enumerate(reader.pages, 1):
            sections.append((obj.extract_text() or "", "", page, 0))
    else:
        raise ValueError("Only Markdown and text PDFs are supported")
    document_id = digest(source)
    chunks = []
    for text, section, page, base in sections:
        cursor = 0
        while cursor < len(text):
            end = min(cursor + chunk_size, len(text))
            # Prefer a whitespace boundary; bounded even for unbroken text.
            if end < len(text):
                split = text.rfind(" ", cursor + chunk_size // 2, end)
                if split != -1:
                    end = split
            raw = text[cursor:end]
            content = raw.strip()
            if content:
                start = base + cursor + len(raw) - len(raw.lstrip())
                stop = base + end - (len(raw) - len(raw.rstrip()))
                key = digest(f"{document_id}:{page}:{section}:{start}:{stop}:{content}")
                chunks.append(Chunk(
                    key, document_id, content, source, title, section, page, start, stop,
                    dict(metadata or {}),
                ))
            if end == len(text):
                break
            cursor = max(cursor + 1, end - overlap)
    if not chunks:
        raise ValueError("No extractable text; scanned PDFs need OCR (not included)")
    return chunks


def ingest_file(path: Path, **kwargs) -> list[Chunk]:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("File exceeds 10 MiB")
    return ingest_bytes(path.read_bytes(), path.name, **kwargs)
