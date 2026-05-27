from langchain_core.embeddings import Embeddings
from llama_cpp import Llama
from typing import List


class JinaEmbeddings(Embeddings):
    """
    Jina Embeddings wrapper using llama-cpp-python for GGUF.
    Code-optimized model for English technical documentation.
    """
    def __init__(self, model_path: str = "./models/jina-embeddings-v2-base-code-f16.gguf"):
        self.model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=8192,
            verbose=False,
            n_threads=10,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.model.create_embedding(text)
            embeddings.append(response["data"][0]["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.model.create_embedding(text)
        return response["data"][0]["embedding"]
