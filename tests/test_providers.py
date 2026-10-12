import json

import httpx
import pytest

from lumina.ingestion import ingest_bytes
from lumina.providers import HTTPProvider


@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_http_provider_parses_quotes_without_network(monkeypatch, name):
    chunk = ingest_bytes(b"Audit logs retention is 30 days.", "policy.md")[0]
    selected = json.dumps([{"chunk_id": chunk.id, "quote": chunk.text}])

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            if name == "openai":
                return {"output": [{"content": [{"type": "output_text", "text": selected}]}],
                        "usage": {"input_tokens": 20, "output_tokens": 10}}
            return {"content": [{"type": "text", "text": selected}],
                    "usage": {"input_tokens": 20, "output_tokens": 10}}

    class Client:
        def __init__(self, timeout):
            assert timeout == 30

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            assert name in url
            assert json["model"] == "demo-model"
            return Response()

    monkeypatch.setattr(httpx, "Client", Client)
    result = HTTPProvider(name, "demo-model", "test-key").select("audit logs retention", [chunk])
    assert result.selections[0].quote == chunk.text
    assert result.input_tokens == 20 and result.output_tokens == 10
