import os
import asyncio
from typing import Optional
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, RequestError
from elastic_transport import AiohttpHttpNode
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from application_logging import ApplicationLogging

logger = ApplicationLogging.depends()


class ProxyAiohttpHttpNode(AiohttpHttpNode):
    """AiohttpHttpNode with proxy support via trust_env=True"""

    def _create_aiohttp_session(self):
        import aiohttp

        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        proxy_url = os.getenv('ELASTIC_HTTPS_PROXY', '').rstrip('/')
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=self.config.connections_per_node,
                verify_ssl=self.config.verify_certs,
            ),
            trust_env=True,  # This enables proxy from environment variables
            loop=self._loop,
            proxy=proxy_url
        )

        logger().debug(f"Aiohttp session created with trust_env=True")


class ESClient:
    def __init__(self, hosts: list[str], api_key: Optional[str] = None, proxy: Optional[str] = None):

        logger().info(f"Initializing ESClient: hosts={hosts}")

        self._es = AsyncElasticsearch(
            hosts,
            api_key=api_key,
            verify_certs=False,
            ssl_show_warn=False,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
            node_class=ProxyAiohttpHttpNode,
        )

    async def __aenter__(self):
        try:
            if not await self._es.ping():
                raise ConnectionError("Elasticsearch ping failed")
            logger().info("Connection established successfully")
            return self
        except Exception as e:
            logger().error(f"Connection error: {type(e).__name__}: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger().error(f"Exception occurred: {exc_type.__name__}: {exc_val}")

        await self._es.close()
        logger().info("Connection closed, proxy restored")

    @retry(
        retry=retry_if_exception_type((ConnectionError, RequestError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3)
    )
    async def search(self, index: str, body: dict, size: int = 50):
        logger().debug(f"Executing search: index={index}, size={size}")
        result = await self._es.search(index=index, body=body, size=size)
        logger().info(f"Search completed: {result['hits']['total']['value']} hits")
        return result
