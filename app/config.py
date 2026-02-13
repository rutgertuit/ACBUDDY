import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


def _get_secret(project_id: str, secret_name: str) -> str:
    """Fetch a secret from Google Cloud Secret Manager."""
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")


@dataclass
class Settings:
    environment: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_webhook_secret: str = ""
    elevenlabs_agent_id: str = ""
    google_cloud_project: str = ""
    google_api_key: str = ""
    deep_max_studies: int = 6
    deep_max_rounds: int = 3
    deep_max_concurrent_studies: int = 3

    def __post_init__(self):
        self.environment = os.getenv("ENVIRONMENT", "local")
        self.google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.deep_max_studies = int(os.getenv("DEEP_MAX_STUDIES", "6"))
        self.deep_max_rounds = int(os.getenv("DEEP_MAX_ROUNDS", "3"))
        self.deep_max_concurrent_studies = int(os.getenv("DEEP_MAX_CONCURRENT_STUDIES", "3"))
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")

        if self.environment == "local":
            self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
            self.elevenlabs_webhook_secret = os.getenv("ELEVENLABS_WEBHOOK_SECRET", "")
            self.elevenlabs_agent_id = os.getenv("ELEVENLABS_AGENT_ID", "")
        else:
            project = self.google_cloud_project
            if not project:
                raise ValueError("GOOGLE_CLOUD_PROJECT must be set in production")
            try:
                self.elevenlabs_api_key = _get_secret(project, "elevenlabs-api-key")
                self.elevenlabs_webhook_secret = _get_secret(project, "elevenlabs-webhook-secret")
                self.elevenlabs_agent_id = _get_secret(project, "elevenlabs-agent-id")
            except Exception:
                logger.exception("Failed to load secrets from Secret Manager")
                raise
