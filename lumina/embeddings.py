"""Stable signed feature hashing. Lexical similarity, not semantic understanding."""
import hashlib
import math
import re

DIMENSIONS = 256
EMBEDDING_MODEL = "hash-bow-v1-256"
STOPWORDS = frozenset(
    "a an the is are was were be been to of for in on at and or with how what when "
    "where who which does do did can it its this that from by as about tell me".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[\w]+", text.casefold()) if t not in STOPWORDS]


class OfflineEmbedder:
    model = EMBEDDING_MODEL
    dimensions = DIMENSIONS

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokens(text):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1 if digest[4] & 1 else -1
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class SemanticEmbedder:
    """Optional local ONNX sentence embeddings; the model is downloaded on first use."""

    model = "BAAI/bge-small-en-v1.5"
    dimensions = 384

    def __init__(self):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("Install the semantic extra: pip install '.[semantic]'") from exc
        self._model = TextEmbedding(model_name=self.model)

    def embed(self, text: str) -> list[float]:
        return list(self._model.passage_embed([text]))[0].tolist()

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.query_embed([text]))[0].tolist()
