"""Grounded quotation answers. Valid citations prove provenance, not factual truth."""
import re
from dataclasses import asdict, dataclass, field

from lumina.embeddings import tokens
from lumina.models import Chunk, Hit

REFUSAL = "I cannot answer from the available document evidence."
ANSWER_POLICY = "quotes-v1"
UNSAFE = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)|"
    r"(?:system|developer)\s*(?:prompt|message|instruction)|"
    r"(?:reveal|exfiltrate|print|send)\b.{0,50}\b(?:secret|password|api.key)|"
    r"(?:you are|act as)\s+(?:an?\s+)?(?:assistant|agent)|"
    r"\[chunk:[^\]]+\]|<\/?(?:system|assistant|tool)>|"
    r"(?:call|execute|run)\s+(?:the\s+)?(?:tool|shell|command)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Selection:
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str
    source: str
    title: str
    section: str
    page: int | None
    start: int
    end: int


@dataclass
class Answer:
    text: str = REFUSAL
    refused: bool = True
    reason: str = "no_evidence"
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def relevant(query: str, quote: str) -> bool:
    query_terms = set(tokens(query))
    shared = query_terms & set(tokens(quote))
    return bool(query_terms) and len(shared) >= min(2, len(query_terms)) and (
        len(shared) / len(query_terms) >= 0.5
    )


def excerpts(query: str, chunks: list[Chunk]) -> list[Selection]:
    result = []
    for chunk in chunks:
        if UNSAFE.search(chunk.text):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.text):
            sentence = sentence.strip()
            if 0 < len(sentence) <= 600 and relevant(query, sentence):
                result.append(Selection(chunk.id, sentence))
    return result


def conflicts(selections: list[Selection]) -> bool:
    """Conservative numeric/negation heuristic; not general contradiction detection."""
    for index, left in enumerate(selections):
        for right in selections[index+1:]:
            if left.chunk_id == right.chunk_id and left.quote == right.quote:
                continue
            words_a = set(tokens(re.sub(r"\b\d+(?:\.\d+)?\b", "", left.quote)))
            words_b = set(tokens(re.sub(r"\b\d+(?:\.\d+)?\b", "", right.quote)))
            # Near-identical subjects with differing values or explicit negation.
            neutral_a = words_a - {"not", "no", "never"}
            neutral_b = words_b - {"not", "no", "never"}
            similarity = len(neutral_a & neutral_b) / max(1, len(neutral_a | neutral_b))
            numbers_a = set(re.findall(r"\b\d+(?:\.\d+)?\b", left.quote))
            numbers_b = set(re.findall(r"\b\d+(?:\.\d+)?\b", right.quote))
            negated_a = bool(words_a & {"not", "no", "never"})
            negated_b = bool(words_b & {"not", "no", "never"})
            if similarity >= 0.75 and (
                (numbers_a and numbers_b and numbers_a != numbers_b)
                or negated_a != negated_b
            ):
                return True
    return False


def eligible(query: str, hits: list[Hit]) -> tuple[list[Chunk], str | None]:
    chunks = [h.chunk for h in hits if not UNSAFE.search(h.chunk.text)]
    evidence = excerpts(query, chunks)
    if not evidence:
        return [], "no_evidence"
    # Inspect the entire retrieved evidence before a provider can cherry-pick it.
    if conflicts(evidence):
        return [], "conflicting_evidence"
    ids = {s.chunk_id for s in evidence}
    return [c for c in chunks if c.id in ids], None


def verify(query: str, selections: list[Selection], chunks: list[Chunk]) -> Answer:
    if not selections or len(selections) > 3:
        return Answer(reason="invalid_provider_output")
    allowed = {c.id: c for c in chunks}
    citations = []
    seen = set()
    for selection in selections:
        chunk = allowed.get(selection.chunk_id)
        if (chunk is None or not selection.quote or len(selection.quote) > 600
                or selection.quote not in chunk.text or UNSAFE.search(selection.quote)
                or not relevant(query, selection.quote)):
            return Answer(reason="invalid_citation")
        identity = (chunk.id, selection.quote)
        if identity in seen:
            continue
        seen.add(identity)
        offset = chunk.text.index(selection.quote)
        citations.append(Citation(
            chunk.id, selection.quote, chunk.source, chunk.title, chunk.section, chunk.page,
            chunk.start + offset, chunk.start + offset + len(selection.quote),
        ))
    if conflicts(selections):
        return Answer(reason="conflicting_evidence")
    text = "\n".join(f'"{c.quote}" [chunk:{c.chunk_id}]' for c in citations)
    return Answer(text, False, "supported_quotes", citations)


def offline_answer(query: str, hits: list[Hit]) -> Answer:
    chunks, reason = eligible(query, hits)
    return Answer(reason=reason) if reason else verify(query, excerpts(query, chunks)[:3], chunks)
