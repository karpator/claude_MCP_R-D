import os

from adapters.es_client import ESClient
from pocketflows import AsyncNode
from application_logging import ApplicationLogging
from .utils import search_multi_index, aggregate_to_documents, rank_documents, SearchConfig

logger = ApplicationLogging.depends()


class RetriveNode(AsyncNode):

    async def prep_async(self, keywords, indices, vector):

        return {
            "keywords": keywords,
            "indices": indices,
            "vector": vector,
        }

    async def exec_async(self, prep_res):

        keywords = prep_res["keywords"]
        vector = prep_res["vector"]
        indices = prep_res["indices"]

        config = SearchConfig(
            indices=indices,
            max_concurrent=6,
            results_per_index=100
        )
        ESCLIENT_HOST = "https://c74719aebcd04d7d8f7803625122a26c.europe-west4.gcp.elastic-cloud.com:443"

        ESCLIENT_API_KEY = "WVc3SXlwUUJvdkp4VmxKU2g2ZmY6MVJQN0pTS2lUYktBVWE1QUhwSmJLZw=="
        try:
            async with ESClient(
                    hosts=[ESCLIENT_HOST],
                    api_key=ESCLIENT_API_KEY,
            ) as es:

                chunks = await search_multi_index(
                    es,
                    keywords=keywords,
                    vectors={"vector_field": vector},
                    config=config
                )

                documents = await aggregate_to_documents(chunks)

                top_docs = rank_documents(documents=documents,
                                          top_n=3,
                                          query_keywords=keywords
                )

                return top_docs

        except Exception as e:
            logger().error(f"ElasticSearch error: {e}")
            return []

    async def post_async(self, shared, prep_res, exec_res):
        shared.context.documents = exec_res
        return "default"
