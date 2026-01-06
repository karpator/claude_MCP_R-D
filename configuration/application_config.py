import os
from typing import Optional

from patterns import singleton
from .configuration import Configuration
from keyvault import GoogleKeyManagementClient


@singleton
class ApplicationConfig(Configuration):
    HOST: str = str(os.environ.get("ApplicationHost", "0.0.0.0"))
    PORT: int = int(os.environ.get("ApplicationPort", "80"))
    API_KEY_AUTHORIZATION_HEADER: str = None

    def __init__(self,
                 google_vault_client: Optional[GoogleKeyManagementClient] = None) -> None:
        super().__init__(google_vault_client)
        self.API_KEY_AUTHORIZATION_HEADER = self._get_key("APIKeyAuthorization")
