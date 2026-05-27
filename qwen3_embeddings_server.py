from langchain_core.embeddings import Embeddings
import requests
from typing import cast, List


class Qwen3EmbeddingsServer(Embeddings):
    """
    Client for the persistent llama-cpp embedding server.
    Much faster than loading the model each time.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")
        self.query_prefix = "Instruct: Given a user query, retrieve relevant documents.\nQuery: "

        # Verify server is reachable
        try:
            requests.get(f"{self.base_url}/health", timeout=10)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach embedding server at {self.base_url}. "
                f"Start it with: python -m llama_cpp.server --model ./models/Qwen3-Embedding-8B-f16.gguf --host 127.0.0.1 --port 8080 --embedding true"
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents RAW - NO PREFIX."""
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": text},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings.append(data["data"][0]["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query WITH instruction prefix."""
        prefixed_text = self.query_prefix + text
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": prefixed_text},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return cast(List[float], data["data"][0]["embedding"])
