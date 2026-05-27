from langchain_core.embeddings import Embeddings
import requests
from typing import List


class StellaEmbeddingsServer(Embeddings):
    """
    Client for the remote Stella embedding server (HTTP).
    Use this when you want to serve Stella embeddings from a separate container/process.
    """

    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url.rstrip("/")
        # Health check
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Stella embedding server at {self.base_url}. "
                f"Start it with: docker-compose up stella-embedding, or run the server manually."
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            f"{self.base_url}/embeddings", json={"input": texts}, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
