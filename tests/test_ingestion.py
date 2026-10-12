import io

import pytest

from lumina.embeddings import OfflineEmbedder
from lumina.ingestion import ingest_bytes
from lumina.store import MemoryStore


def test_markdown_metadata_offsets_and_determinism():
    raw = "# Handbook\n\n## Retention\nLogs are retained for 30 days.\n## Security\nUse TLS."
    chunks = ingest_bytes(raw.encode(), "handbook.md", {"team": "demo"})
    assert chunks == ingest_bytes(raw.encode(), "handbook.md", {"team": "demo"})
    assert {c.section for c in chunks} == {"Retention", "Security"}
    assert all(raw[c.start:c.end] == c.text for c in chunks)
    assert all(c.metadata == {"team": "demo"} for c in chunks)
    assert all(c.title == "Handbook" for c in chunks)


def test_pdf_page_provenance():
    pytest.importorskip("pypdf")
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    output = io.BytesIO()
    pdf = canvas.Canvas(output)
    pdf.setTitle("Policy")
    for text in ("Logs are retained for 30 days.", "Support opens at 09:00."):
        pdf.drawString(50, 750, text)
        pdf.showPage()
    pdf.save()
    chunks = ingest_bytes(output.getvalue(), "policy.pdf")
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].title == "Policy"
    assert "30 days" in chunks[0].text


def test_blank_pdf_and_unsupported_files():
    writer = pytest.importorskip("pypdf").PdfWriter()
    writer.add_blank_page(width=100, height=100)
    data = io.BytesIO()
    writer.write(data)
    with pytest.raises(ValueError, match="No extractable"):
        ingest_bytes(data.getvalue(), "scan.pdf")


def test_unsupported_files():
    with pytest.raises(ValueError, match="Only Markdown"):
        ingest_bytes(b"hello", "file.exe")


def test_chunk_limits_and_overlap():
    raw = "alpha beta gamma " * 100
    chunks = ingest_bytes(raw.encode(), "long.md", chunk_size=100, overlap=20)
    assert len(chunks) > 5
    assert all(len(c.text) <= 100 and raw[c.start:c.end] == c.text for c in chunks)
    assert chunks[-1].end == len(raw.rstrip())


def test_embeddings_stable_normalized_and_replacement():
    embed = OfflineEmbedder()
    vec = embed.embed("retention logs 30 days")
    assert len(vec) == 256
    assert vec == embed.embed("retention logs 30 days")
    assert sum(v*v for v in vec) == pytest.approx(1)
    store = MemoryStore()
    chunks = ingest_bytes(b"Logs retained for 30 days.", "policy.md")
    assert store.replace(chunks, [embed.embed(c.text) for c in chunks])
    assert not store.replace(chunks, [embed.embed(c.text) for c in chunks])
    newer = ingest_bytes(b"Logs retained for 60 days.", "policy.md")
    assert store.replace(newer, [embed.embed(c.text) for c in newer])
    assert store.get(chunks[0].id) is None
    assert store.revision() == 2
