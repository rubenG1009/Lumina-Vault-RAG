from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    source: str
    title: str
    section: str = ""
    page: int | None = None
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {"chunk": self.chunk.to_dict(), "score": self.score}
