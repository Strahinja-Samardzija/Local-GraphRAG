#!/usr/bin/env python3
"""
RAG retrieval executed by Qwen Code SubAgent.
Supports multiple collections (qwen3, jinaai, etc.)
"""

import argparse
import json
from urllib.parse import urlparse

from langchain_chroma import Chroma
from qwen3_embeddings import Qwen3Embeddings
from qwen3_embeddings_server import Qwen3EmbeddingsServer
from jina_embeddings import JinaEmbeddings
from stella_embeddings import StellaEmbeddings

CHROMA_PERSIST_DIR = "./chroma_db"
SEARCH_K = 5
DEFAULT_COLLECTION = "qwen3"


def _make_vectorstore(persist_directory=None, host=None, port=None,
                      embedding_function=None, collection_name=None):
    """Factory that works with either local persist or remote ChromaDB."""
    kwargs = {}
    if embedding_function:
        kwargs["embedding_function"] = embedding_function
    if collection_name:
        kwargs["collection_name"] = collection_name

    if host:
        kwargs["host"] = host
        kwargs["port"] = port or 8000
        return Chroma(**kwargs)
    else:
        kwargs["persist_directory"] = persist_directory or CHROMA_PERSIST_DIR
        return Chroma(**kwargs)


def get_collection(host=None, port=None, collection_name=None):
    """Get direct access to the Chroma collection for metadata queries."""
    import chromadb
    col_name = collection_name or DEFAULT_COLLECTION
    if host:
        client = chromadb.HttpClient(host=host, port=port or 8000)
    else:
        from langchain_chroma import Chroma
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, collection_name=col_name)
        client = vectorstore._client
    return client.get_or_create_collection(name=col_name)


def get_context_chunks(source_file: str, chunk_index: int, context: int,
                       host=None, port=None, collection_name=None):
    """Retrieve chunks before and after the given chunk_index within the same source file."""
    if context <= 0:
        return []

    collection = get_collection(host=host, port=port, collection_name=collection_name)
    max_index = chunk_index + context
    min_index = chunk_index - context

    results = collection.get(
        where={"source_file": source_file},
        include=["metadatas", "documents"]
    )

    context_chunks = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        idx = meta.get("chunk_index", -1)
        if min_index <= idx <= max_index and idx != chunk_index:
            context_chunks.append({
                "content": doc,
                "chunk_index": idx,
                "offset": idx - chunk_index
            })

    context_chunks.sort(key=lambda x: x["chunk_index"])
    return context_chunks


def search_query(query: str, k: int = SEARCH_K, context: int = 0,
                 use_server: bool = False, server_url: str = "http://127.0.0.1:8080",
                 chroma_url: str = None, collection_name: str = None):
    """
    Search the vector store for similar documents.

    Args:
        query: The search query
        k: Number of results to return
        context: Number of neighbouring chunks to include before/after each match
        use_server: Use persistent embedding server
        server_url: URL of the embedding server
        chroma_url: URL of remote ChromaDB server (overrides local persist)
        collection_name: ChromaDB collection name (default: qwen3)
    """
    collection = collection_name or DEFAULT_COLLECTION

    # Auto-select embedding model based on collection name
    if collection == "jinaai":
        embeddings = JinaEmbeddings()
    elif collection.startswith("stella"):
        embeddings = StellaEmbeddings()
    elif use_server:
        embeddings = Qwen3EmbeddingsServer(base_url=server_url)
    else:
        embeddings = Qwen3Embeddings()

    # Parse ChromaDB remote URL if provided
    chroma_host, chroma_port = None, None
    if chroma_url:
        parsed = urlparse(chroma_url)
        chroma_host = parsed.hostname or parsed.path
        chroma_port = parsed.port or 8000

    vectorstore = _make_vectorstore(
        persist_directory=CHROMA_PERSIST_DIR,
        host=chroma_host,
        port=chroma_port,
        embedding_function=embeddings,
        collection_name=collection,
    )
    results = vectorstore.similarity_search_with_score(query, k=k)

    output = []
    for doc, score in results:
        result = {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "language": doc.metadata.get("language", "unknown"),
            "chunk_type": doc.metadata.get("node_type", "original_content"),
            "cluster_id": doc.metadata.get("cluster_id", -1),
        }

        if context > 0:
            source_file = doc.metadata.get("source_file", "unknown")
            chunk_index = doc.metadata.get("chunk_index", -1)
            if chunk_index >= 0:
                result["context_chunks"] = get_context_chunks(
                    source_file, chunk_index, context,
                    host=chroma_host, port=chroma_port, collection_name=collection_name,
                )

        output.append(result)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG retrieval pipeline")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("-k", "--k", type=int, default=SEARCH_K, help="Number of results")
    parser.add_argument("-c", "--context", type=int, default=0,
                        help="Number of neighboring chunks to include before/after each match")
    parser.add_argument("--server", action="store_true", help="Use persistent embedding server")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080",
                        help="URL of embedding server (default: http://127.0.0.1:8080)")
    parser.add_argument("--chroma-url", default=None,
                        help="URL of remote ChromaDB server (overrides local persist)")
    parser.add_argument("--collection", default=None,
                        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})")
    args = parser.parse_args()

    results = search_query(
        args.query, k=args.k, context=args.context,
        use_server=args.server, server_url=args.server_url,
        chroma_url=args.chroma_url, collection_name=args.collection,
    )

    if args.json:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    else:
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['source']} (score: {r['score']:.3f}, lang: {r['language']})")

            # Print context chunks before
            if "context_chunks" in r:
                before = [c for c in r["context_chunks"] if c["offset"] < 0]
                for ctx in before:
                    print(f"  [−{abs(ctx['offset'])}] {ctx['content'][:200]}...")

            # Print main chunk
            print(f"  >>> {r['content'][:300]}...")

            # Print context chunks after
            if "context_chunks" in r:
                after = [c for c in r["context_chunks"] if c["offset"] > 0]
                for ctx in after:
                    print(f"  [+{ctx['offset']}] {ctx['content'][:200]}...")

            print()
