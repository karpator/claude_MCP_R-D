from typing import List, Optional, Dict
import numpy as np
import logging
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core import exceptions
from typing import Any
import vertexai
from vertexai.language_models import TextEmbeddingModel



class EmbeddingService:
    """Optimized Embedding Service with caching and parallel processing"""
    _cache: Dict[str, List[float]] = {}

    def __init__(self, model: TextEmbeddingModel):
        """Initialize with preconfigured model"""
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.logger.info("Embedding service initialized")

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> List[float]:
        return self._get_embedding_with_retry(text)

    @retry(
        retry=retry_if_exception_type(exceptions.ResourceExhausted),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3)
    )
    def _get_embedding_with_retry(self, text: str) -> List[float]:
        try:
            embedding = self.model.get_embeddings([text])[0]
            return embedding.values
        except Exception as e:
            if isinstance(e, exceptions.ResourceExhausted):
                self.logger.warning("Quota exceeded, retrying...")
                raise
            self.logger.error(f"Error getting embedding: {str(e)}")
            raise

    async def get_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Async batch processing of embeddings"""
        embeddings = []

        async def process_batch(batch):
            tasks = []
            for text in batch:
                if text in self._cache:
                    tasks.append(self._cache[text])
                else:
                    task = self.executor.submit(self._get_cached_embedding, text)
                    tasks.append(task)

            results = await asyncio.gather(
                *[asyncio.wrap_future(t) if not isinstance(t, list) else asyncio.sleep(0, result=t) for t in tasks])
            return results

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        for batch in batches:
            batch_embeddings = await process_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def get_embedding(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]

        embedding = self._get_cached_embedding(text)
        self._cache[text] = embedding
        return embedding

    def calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        try:
            v1, v2 = np.array(vector1), np.array(vector2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            raise
