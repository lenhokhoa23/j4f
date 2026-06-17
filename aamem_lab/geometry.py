from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .schemas import MemoryRecord
from .text import cosine_counter, tokenize


@dataclass(frozen=True)
class GeometrySignal:
    memory_id: str
    cluster_id: int
    centrality: float
    novelty: float
    nearest_neighbor_id: str | None
    nearest_neighbor_score: float


def compute_local_geometry(records: Sequence[MemoryRecord], cluster_threshold: float = 0.18) -> Dict[str, GeometrySignal]:
    """Greedy token-vector clustering for Phase 3 geometry diagnostics.

    This is deliberately simple and deterministic. It is not presented as a
    final manifold method; it exposes cluster/novelty signals that can later be
    replaced by embedding-based clustering.
    """

    vectors = {r.id: Counter(tokenize(r.content)) for r in records}
    clusters: List[List[str]] = []
    for record in records:
        best_cluster = None
        best_score = -1.0
        for idx, cluster in enumerate(clusters):
            centroid = Counter()
            for mid in cluster:
                centroid.update(vectors[mid])
            score = cosine_counter(vectors[record.id], centroid)
            if score > best_score:
                best_score = score
                best_cluster = idx
        if best_cluster is None or best_score < cluster_threshold:
            clusters.append([record.id])
        else:
            clusters[best_cluster].append(record.id)

    cluster_of = {mid: idx for idx, cluster in enumerate(clusters) for mid in cluster}
    signals: Dict[str, GeometrySignal] = {}
    for record in records:
        rid = record.id
        cluster = clusters[cluster_of[rid]]
        nearest_id = None
        nearest_score = 0.0
        for other in records:
            if other.id == rid:
                continue
            score = cosine_counter(vectors[rid], vectors[other.id])
            if score > nearest_score:
                nearest_id = other.id
                nearest_score = score
        if len(cluster) <= 1:
            centrality = 0.0
        else:
            centrality = sum(
                cosine_counter(vectors[rid], vectors[mid])
                for mid in cluster
                if mid != rid
            ) / max(1, len(cluster) - 1)
        novelty = 1.0 - nearest_score
        signals[rid] = GeometrySignal(
            memory_id=rid,
            cluster_id=cluster_of[rid],
            centrality=centrality,
            novelty=novelty,
            nearest_neighbor_id=nearest_id,
            nearest_neighbor_score=nearest_score,
        )
    return signals

