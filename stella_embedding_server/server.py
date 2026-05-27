#!/usr/bin/env python3
"""
Standalone HTTP server for Stella embeddings (ONNX).
Lazy-loads the model on first request. Exposes OpenAI-compatible /embeddings endpoint.
"""

import sys, os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from stella_embeddings import StellaEmbeddings

app = FastAPI()
_model = None


def get_model():
    global _model
    if _model is None:
        print("[Stella Server] Loading ONNX model (lazy)...")
        _model = StellaEmbeddings()
        print("[Stella Server] Model loaded and ready.")
    return _model


class EmbeddingRequest(BaseModel):
    input: List[str]


@app.post("/embeddings")
def embeddings(req: EmbeddingRequest):
    """Return embeddings for the given texts."""
    model = get_model()
    vectors = model.embed_documents(req.input)
    return {"data": [{"embedding": v} for v in vectors]}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
