import os

from patterns import singleton
from .types import AuthenticationMethodTypes
from google.auth import (
    compute_engine,
    default
)
from google.auth.credentials import Credentials
from google.oauth2 import service_account
import google.auth.transport.requests


@singleton
class GoogleKeyManagementConfig:
    PROJECT_ID: str = str(os.getenv("GoogleProjectId", "hu0092-bus-t-ai"))  # "custom-casing-401908"))
    AUTHENTICATION_METHOD: str = str(
        os.getenv("GoogleAuthenticationMethod", AuthenticationMethodTypes.APPLICATION_DEFAULT))
    SERVICE_ACCOUNT_CREDENTIALS: Credentials = None

    def __init__(self) -> None:
        """
        Constructor for GoogleKeyManagementConfig class that initializes the service account credentials.

        :return: None
        """

        match self.AUTHENTICATION_METHOD:
            case AuthenticationMethodTypes.COMPUTE_ENGINE:
                self.SERVICE_ACCOUNT_CREDENTIALS = compute_engine.Credentials()
            case AuthenticationMethodTypes.SERVICE_ACCOUNT_JSON:
                file_root_path = os.path.abspath(__file__).__str__().split("configuration")[0]
                key_path = "configuration/google_account_key.json"
                google_key_path = file_root_path + key_path
                self.SERVICE_ACCOUNT_CREDENTIALS = service_account.Credentials.from_service_account_file(
                    google_key_path)
            case AuthenticationMethodTypes.APPLICATION_DEFAULT:
                credentials, project_id = default()
                self.SERVICE_ACCOUNT_CREDENTIALS = credentials
            case _:
                raise ValueError("Invalid authentication method")

        request = google.auth.transport.requests.Request()
        if self.SERVICE_ACCOUNT_CREDENTIALS:
            self.SERVICE_ACCOUNT_CREDENTIALS.refresh(request)
