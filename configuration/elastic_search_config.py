import os
from typing import List

from .configuration import Configuration
from keyvault.google_key_management_client import GoogleKeyManagementClient


class ElasticSearchConfig(Configuration):
    ELASTIC_SEARCH_API: List[str] = [
        "https://hu0092-bus-t-ai-es01.es.europe-west4.gcp.elastic-cloud.com"
    ]
    ENVIRONMENT: str = str(os.getenv("Environment", "dev"))
    ELASTIC_SEARCH_API_KEY: str = None
    ELASTIC_PROXI_API: List[str] = []

    def __init__(
            self,
            google_key_management_client: GoogleKeyManagementClient = None
    ):
        super().__init__(google_key_management_client)
        self.ELASTIC_SEARCH_API_KEY = self._get_key("ElasticSearch_API_KEY")
