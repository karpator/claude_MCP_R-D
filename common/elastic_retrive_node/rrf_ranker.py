import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from enum import Enum
from rapidfuzz import fuzz
from application_logging import ApplicationLogging
from .models import DocumentCandidate

logger = ApplicationLogging.depends()


class TemporalStrategy(Enum):
    """Temporal keyword scoring strategies"""
    DISABLED = "disabled"
    INTERACTION = "interaction"
    WEIGHTED = "weighted"
    STRICT = "strict"


@dataclass
class RRFConfig:
    """Reciprocal Rank Fusion configuration"""
    k: int = 60
    agreement_boost: float = 0.3
    query_overlap_weight: float = 0.2
    min_overlap_threshold: float = 0.3
    fuzzy_threshold: int = 85
    min_token_coverage: float = 0.5
    temporal_weight: float = 0.15
    temporal_strategy: TemporalStrategy = TemporalStrategy.INTERACTION
    year_pattern: str = r'\b(19|20)\d{2}\b'


class RRFScorer:
    """Production RRF scorer with agreement, query overlap, and temporal bonuses"""

    def __init__(self, config: RRFConfig = None):
        self.config = config or RRFConfig()
        self._year_regex = re.compile(self.config.year_pattern)
        self._doc_cache = {}

    def score_documents(
        self,
        documents: list[DocumentCandidate],
        query_keywords: list[str] = None
    ) -> list[tuple[DocumentCandidate, float, Dict]]:
        """Score documents and return (doc, score, debug_info) tuples"""
        self._doc_cache.clear()
        scored = []

        for doc in documents:
            doc_text, doc_words = self._get_cached_context(doc)

            base = self._base_rrf(doc)
            agreement = self._agreement_bonus(doc)
            overlap = self._query_overlap_cached(doc_words, query_keywords) if query_keywords else 0.0
            temporal = self._temporal_bonus(doc_text, doc_words, query_keywords) if query_keywords else 0.0

            final_score = base + agreement + overlap + temporal
            debug_info = {
                "base_rrf": base,
                "agreement": agreement,
                "overlap": overlap,
                "temporal": temporal
            }
            scored.append((doc, final_score, debug_info))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _get_cached_context(self, doc: DocumentCandidate) -> tuple[str, list[str]]:
        """Cache document text parsing per scoring cycle"""
        doc_id = id(doc)
        if doc_id not in self._doc_cache:
            text = " ".join(c.content.lower() for c in doc.chunks)
            self._doc_cache[doc_id] = (text, text.split())
        return self._doc_cache[doc_id]

    def _base_rrf(self, doc: DocumentCandidate) -> float:
        """Core RRF: sum of 1/(k + rank) for each search type"""
        if not doc.chunks:
            return 0.0

        score = 0.0
        by_type = defaultdict(list)

        for chunk in doc.chunks:
            by_type[chunk.search_type].append(chunk)

        for chunks in by_type.values():
            sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
            for rank, chunk in enumerate(sorted_chunks, start=1):
                score += 1.0 / (self.config.k + rank)

        return score

    def _agreement_bonus(self, doc: DocumentCandidate) -> float:
        """Bonus when same chunk appears in multiple search types"""
        if not doc.chunks:
            return 0.0

        chunk_signatures = defaultdict(set)

        for chunk in doc.chunks:
            signature = f"{chunk.metadata.page_number}_{chunk.metadata.chunk_number}"
            chunk_signatures[signature].add(chunk.search_type)

        cross_method_count = sum(
            1 for types in chunk_signatures.values() if len(types) > 1
        )

        agreement_ratio = cross_method_count / len(doc.chunks)
        return agreement_ratio * self.config.agreement_boost

    def _query_overlap_cached(self, doc_words: list[str], keywords: list[str]) -> float:
        """Query overlap bonus using pre-cached doc words"""
        if not keywords:
            return 0.0

        total_score = sum(
            self._keyword_match_score(kw.lower(), doc_words)
            for kw in keywords
        )
        overlap_ratio = total_score / len(keywords)

        if overlap_ratio < self.config.min_overlap_threshold:
            return 0.0

        return overlap_ratio * self.config.query_overlap_weight

    def _keyword_match_score(self, keyword: str, doc_words: list[str]) -> float:
        """
        Calculate match score for a single keyword using fuzzy token matching.

        Single-word: binary match (1.0 or 0.0)
        Multi-word: gradient score based on token coverage (0.0-1.0)

        Returns 0.0 if coverage below min_token_coverage threshold.
        """
        tokens = keyword.split()

        if not tokens:
            return 0.0

        if len(tokens) == 1:
            return 1.0 if any(
                fuzz.ratio(keyword, word) >= self.config.fuzzy_threshold
                for word in doc_words
            ) else 0.0

        matched = sum(
            1 for token in tokens
            if any(fuzz.ratio(token, word) >= self.config.fuzzy_threshold for word in doc_words)
        )

        coverage = matched / len(tokens)
        return coverage if coverage >= self.config.min_token_coverage else 0.0

    def _temporal_bonus(self, doc_text: str, doc_words: list[str], keywords: list[str]) -> float:
        """Temporal keyword bonus - auto-skips if no years in keywords"""
        if not keywords:
            return 0.0

        temporal_kws, non_temporal_kws = self._split_temporal_keywords(keywords)

        if not temporal_kws:
            return 0.0

        if self.config.temporal_strategy == TemporalStrategy.DISABLED:
            return 0.0

        if not non_temporal_kws:
            return 0.0

        doc_years = set(self._year_regex.findall(doc_text))
        has_temporal_match = any(year in doc_years for year in temporal_kws)

        if not has_temporal_match:
            return 0.0

        return self._apply_temporal_strategy(non_temporal_kws, doc_words)

    def _split_temporal_keywords(self, keywords: list[str]) -> tuple[list[str], list[str]]:
        """Split keywords into temporal (years) and non-temporal"""
        temporal = [kw for kw in keywords if self._year_regex.fullmatch(kw.strip())]
        non_temporal = [kw for kw in keywords if kw not in temporal]
        return temporal, non_temporal

    def _apply_temporal_strategy(self, non_temporal_kws: list[str], doc_words: list[str]) -> float:
        """Apply configured temporal scoring strategy"""
        scores = [
            self._keyword_match_score(kw.lower(), doc_words)
            for kw in non_temporal_kws
        ]

        strategy = self.config.temporal_strategy

        if strategy == TemporalStrategy.INTERACTION:
            coverage = sum(scores) / len(scores)
            return coverage * self.config.temporal_weight

        elif strategy == TemporalStrategy.WEIGHTED:
            return self.config.temporal_weight if any(s > 0 for s in scores) else 0.0

        elif strategy == TemporalStrategy.STRICT:
            return self.config.temporal_weight if all(s >= 0.5 for s in scores) else 0.0

        return 0.0


def rank_documents(
        documents: list[DocumentCandidate],
        top_n: int = 50,
        query_keywords: list[str] = None,
        config: RRFConfig = None
) -> list[DocumentCandidate]:
    """Rank documents using Reciprocal Rank Fusion with enhancements"""

    scorer = RRFScorer(config)
    scored = scorer.score_documents(documents, query_keywords)

    top_scores = [f"{score:.4f}" for _, score, _ in scored[:3]]
    logger().info(f"RRF ranking completed: {len(documents)} docs, top 3 scores: {top_scores}")

    return [doc for doc, _, _ in scored[:top_n]]
