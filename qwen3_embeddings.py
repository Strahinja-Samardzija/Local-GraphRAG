from langchain_core.embeddings import Embeddings
from llama_cpp import Llama
from typing import cast, List


class Qwen3Embeddings(Embeddings):
    """
    Qwen3-Embedding-8B wrapper using llama-cpp-python for GGUF.
    Enforces Asymmetric Retrieval:
    - Documents: RAW (no prefix)
    - Queries: Prefixed with instruction
    """

    def __init__(self, model_path: str = "./models/Qwen3-Embedding-8B-f16.gguf"):
        # Initialize Llama.cpp with embedding mode enabled
        self.model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=8192,  # Supports large context windows natively
            verbose=False,
            n_threads=10  # Optimized for Ryzen 5 7600 (12 threads total)
        )
        self.query_prefix = "Instruct: Given a user query, retrieve relevant documents.\nQuery: "

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents RAW - NO PREFIX."""
        embeddings = []
        for text in texts:
            response = self.model.create_embedding(text)
            embeddings.append(response["data"][0]["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query WITH instruction prefix."""
        prefixed_text = self.query_prefix + text
        response = self.model.create_embedding(prefixed_text)
        
        # Extract the raw embedding
        embedding = response["data"][0]["embedding"]
        
        # Use cast to reassure Pylance that it is strictly a List[float]
        return cast(List[float], embedding)
