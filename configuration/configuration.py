from abc import ABC
from typing import Optional

from keyvault import GoogleKeyManagementClient


class Configuration(ABC):
    def __init__(self,
                 key_vault_client: Optional[GoogleKeyManagementClient]) -> None:
        if key_vault_client is None:
            key_vault_client = GoogleKeyManagementClient()
        self._key_vault_client = key_vault_client

    def _get_key(self, key: str) -> str:
        return str(self._key_vault_client.get_key(key))
