from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

from .schemas import Candidate, MemoryRecord
from .text import cosine_counter, jaccard, tokenize


class BM25Retriever:
    """Small deterministic BM25 retriever with no external dependencies."""

    def __init__(self, records: Sequence[MemoryRecord], k1: float = 1.5, b: float = 0.75):
        self.records = list(records)
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = [tokenize(r.content) for r in self.records]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.term_freqs = [Counter(toks) for toks in self.doc_tokens]
        df: Dict[str, int] = defaultdict(int)
        for toks in self.doc_tokens:
            for tok in set(toks):
                df[tok] += 1
        n = max(1, len(self.records))
        self.idf = {
            tok: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def search(self, query: str, k: int = 5) -> List[Candidate]:
        q_terms = tokenize(query)
        scored = []
        for idx, record in enumerate(self.records):
            score = self._score(q_terms, idx)
            scored.append((score, idx))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            Candidate(memory=self.records[idx], score=float(score), rank=rank + 1, source="bm25")
            for rank, (score, idx) in enumerate(scored[:k])
        ]

    def _score(self, q_terms: Iterable[str], doc_idx: int) -> float:
        tf = self.term_freqs[doc_idx]
        dl = self.doc_len[doc_idx]
        denom_norm = self.k1 * (1.0 - self.b + self.b * dl / max(1e-9, self.avgdl))
        score = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            numerator = freq * (self.k1 + 1.0)
            score += self.idf.get(term, 0.0) * numerator / (freq + denom_norm)
        return score


def build_token_knn_links(
    records: Sequence[MemoryRecord],
    top_n: int = 3,
    min_jaccard: float = 0.04,
) -> Dict[str, List[str]]:
    """Build cheap local-neighbor links to mimic A-MEM neighborhood expansion."""

    token_sets = {r.id: set(tokenize(r.content)) for r in records}
    links: Dict[str, List[str]] = {}
    for record in records:
        scored = []
        rtoks = token_sets[record.id]
        for other in records:
            if other.id == record.id:
                continue
            score = jaccard(rtoks, token_sets[other.id])
            if score >= min_jaccard:
                scored.append((score, other.id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        links[record.id] = [mid for _, mid in scored[:top_n]]
    return links


def cosine_query_scores(query: str, records: Sequence[MemoryRecord]) -> Dict[str, float]:
    q = Counter(tokenize(query))
    return {r.id: cosine_counter(q, Counter(tokenize(r.content))) for r in records}

