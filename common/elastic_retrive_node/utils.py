import asyncio
from collections import defaultdict
from dataclasses import dataclass
from adapters.es_client import ESClient
from .models import SearchResult, ChunkMetadata, DocumentCandidate
from .rrf_ranker import RRFScorer, RRFConfig
from application_logging import ApplicationLogging
import re

logger = ApplicationLogging.depends()

@dataclass
class SearchConfig:
    """Simple search configuration"""
    indices: list[str]
    max_concurrent: int = 6
    results_per_index: int = 50
    min_score: float = 0.1


def lexical_query(keywords: list[str], min_score: float = 0.5) -> dict:
    """Production lexical search - performance optimized for massive datasets"""

    should = [
        # Exact
        *[{"term": {"content.enum": {"value": kw.lower(), "boost": 5.0}}}
          for kw in keywords],

        # Stem + fuzzy
        *[{"match": {"content.stem": {
            "query": kw.lower(),
            "boost": 3.0,
            "fuzziness": "AUTO",
            "prefix_length": 2,
            "max_expansions": 50
        }}} for kw in keywords],

        # Base fallback
        *[{"match": {"content": {"query": kw.lower(), "boost": 1.5}}}
          for kw in keywords],

        # 2-gram phrases
        *[{"match_phrase": {"content": {"query": " ".join(keywords[i:i + 2]), "boost": 4.0}}}
          for i in range(len(keywords) - 1)]
    ]

    return {
        "query": {"bool": {
            "should": should,
            "minimum_should_match": max(1, len(keywords) // 2)
        }},
        "_source": ["content", "id", "metadata.pdf_name", "metadata.gcs_uri"],
        "min_score": min_score
    }

def vector_query(field: str, vector: list[float], k: int = 50) -> dict:
    """Build vector query"""
    return {
        "knn": {
            "field": field,
            "query_vector": vector,
            "k": k,
            "num_candidates": max(k * 2, 200)
        },
        "_source": ["content", "id", "metadata.pdf_name", "metadata.gcs_uri"]
    }


async def search_multi_index(
        es: ESClient,
        keywords: list[str],
        vectors: dict[str, list[float]],
        config: SearchConfig
) -> list[SearchResult]:
    """
    Execute parallel multi-index hybrid search

    Returns list of SearchResult (chunks) from all indices
    """

    tasks = [
        *[
            _search_task(es, idx, lexical_query(keywords, config.min_score), "lexical", config.results_per_index)
            for idx in config.indices
            if keywords
        ],
        *[
            _search_task(es, idx, vector_query(field, vec), f"semantic_{field}", config.results_per_index)
            for idx in config.indices
            for field, vec in vectors.items()
        ]
    ]

    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def bounded(task):
        async with semaphore:
            return await task

    results = await asyncio.gather(*[bounded(t) for t in tasks], return_exceptions=True)

    chunks = [
        chunk
        for result in results
        if not isinstance(result, Exception)
        for chunk in result
    ]

    logger().info(f"search_completed: Retrieved {len(chunks)} chunks from {len(config.indices)} indices.")
    return chunks


async def _search_task(
        es: ESClient,
        index: str,
        query: dict,
        search_type: str,
        size: int
) -> list[SearchResult]:
    """Execute single search and convert to SearchResult"""

    response = await es.search(index, query, size)

    return [
        SearchResult(
            content=hit["_source"].get("content", ""),
            metadata=ChunkMetadata(**hit["_source"].get("metadata", {})),
            score=hit["_score"],
            source_index=index,
            search_type=search_type,
            vector_field=search_type.split("_")[1] if "semantic" in search_type else None
        )
        for hit in response.get("hits", {}).get("hits", [])
    ]


async def aggregate_to_documents(chunks: list[SearchResult]) -> list[DocumentCandidate]:

    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk.metadata.pdf_name].append(chunk)

    async def build_candidate(pdf_name, chunks):
        full_content = await get_full_content(pdf_name, chunks[0].source_index)
        pdf_gcs_uri = next((re.sub(r'_page_\\d+', '', c.metadata.gcs_uri) for c in chunks if c.metadata.gcs_uri), None)
        return DocumentCandidate(
            document_id=pdf_name,
            chunks=chunks,
            full_content=full_content,
            pdf_gcs_uri=pdf_gcs_uri
        )

    coros = [build_candidate(pdf_name, chunks) for pdf_name, chunks in groups.items()]
    return await asyncio.gather(*coros)


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

async def get_full_content(pdf_name: str, source_index: str) -> str | None:
    ESCLIENT_HOST = "https://c74719aebcd04d7d8f7803625122a26c.europe-west4.gcp.elastic-cloud.com:443"

    ESCLIENT_API_KEY = "WVc3SXlwUUJvdkp4VmxKU2g2ZmY6MVJQN0pTS2lUYktBVWE1QUhwSmJLZw=="
    async with ESClient(
            hosts=ESCLIENT_HOST,
            api_key=ESCLIENT_API_KEY,
    ) as es:
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"term": {"metadata.pdf_name.keyword": {"value": pdf_name}}},
                        {"term": {"metadata.pdf_name.enum": {"value": pdf_name}}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["metadata.all_md_pages"]
        }

        response = await es.search(index=source_index, body=query, size=1)
        hits = response.get("hits", {}).get("hits", [])
        if hits:
            metadata = hits[0]["_source"].get("metadata", {})
            return metadata.get("all_md_pages")
    return None