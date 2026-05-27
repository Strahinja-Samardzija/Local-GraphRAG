#!/usr/bin/env python3
"""
Wrapper script to run the MCP server with environment-based configuration.
This allows the MCP server to be run without CLI arguments.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set defaults from environment or use defaults
os.environ.setdefault("CHROMA_URL", os.getenv("CHROMA_URL", "http://localhost:8000"))
os.environ.setdefault(
    "EMBEDDING_URL", os.getenv("EMBEDDING_URL", "http://localhost:8080")
)
os.environ.setdefault("COLLECTION", os.getenv("COLLECTION", "stella_contextual"))
os.environ.setdefault(
    "STELLA_EMBEDDING_URL", os.getenv("STELLA_EMBEDDING_URL", "http://localhost:8081")
)
os.environ.setdefault("NEO4J_URI", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
os.environ.setdefault("NEO4J_USER", os.getenv("NEO4J_USER", "neo4j"))
os.environ.setdefault("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "your_password"))

# Import and run the MCP server
from mcp.server.fastmcp import FastMCP
import chromadb
from langchain_chroma import Chroma
import json
from typing import Optional

# Import embedding functions
try:
    from stella_embeddings import StellaEmbeddings
    from jina_embeddings import JinaEmbeddings

    _HAS_LOCAL_EMBEDDINGS = True
except ImportError:
    _HAS_LOCAL_EMBEDDINGS = False

try:
    from stella_embedding_server import StellaEmbeddingsServer

    _HAS_STELLA_SERVER = True
except ImportError:
    _HAS_STELLA_SERVER = False

from qwen3_embeddings_server import Qwen3EmbeddingsServer
from neo4j import GraphDatabase

# Configuration
CHROMA_URL = os.environ["CHROMA_URL"]
EMBEDDING_URL = os.environ["EMBEDDING_URL"]
COLLECTION = os.environ["COLLECTION"]
DEFAULT_K = 5
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


def _parse_chroma_url(url: str) -> tuple:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or parsed.path or "localhost"
    port = parsed.port or 8000
    return host, port


def _get_vectorstore(collection_name: Optional[str] = None) -> Chroma:
    col = collection_name or COLLECTION

    if col == "jinaai":
        if not _HAS_LOCAL_EMBEDDINGS:
            raise RuntimeError("Jina embeddings require local dependencies.")
        embeddings = JinaEmbeddings()
    elif col.startswith("stella"):
        # Always use local Stella embeddings to ensure consistent results
        # The stella_contextual collection was created with local embeddings
        if not _HAS_LOCAL_EMBEDDINGS:
            raise RuntimeError(
                "Stella embeddings require local dependencies."
            )
        embeddings = StellaEmbeddings()
    else:
        embeddings = Qwen3EmbeddingsServer(base_url=EMBEDDING_URL)

    host, port = _parse_chroma_url(CHROMA_URL)
    return Chroma(
        host=host,
        port=port,
        collection_name=col,
        embedding_function=embeddings,
    )


def _get_context_chunks(
    source_file: str,
    chunk_index: int,
    context: int,
    collection_name: Optional[str] = None,
):
    """Retrieve neighbouring chunks within the same source file."""
    if context <= 0:
        return []

    host, port = _parse_chroma_url(CHROMA_URL)
    client = chromadb.HttpClient(host=host, port=port)
    col = client.get_or_create_collection(name=collection_name or COLLECTION)
    max_index = chunk_index + context
    min_index = chunk_index - context

    results = col.get(
        where={"source_file": source_file},
        include=["metadatas", "documents"],
    )

    ctx = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        idx = meta.get("chunk_index", -1)
        if min_index <= idx <= max_index and idx != chunk_index:
            ctx.append(
                {
                    "content": doc,
                    "chunk_index": idx,
                    "offset": idx - chunk_index,
                }
            )

    ctx.sort(key=lambda x: x["chunk_index"])
    return ctx


# Create MCP server
mcp = FastMCP(
    "sota-rag",
    instructions="RAG retrieval over a document corpus using embeddings, with graph query support",
)


@mcp.tool()
def rag_query(
    query: str,
    k: int = DEFAULT_K,
    context: int = 0,
    collection: Optional[str] = None,
) -> str:
    """Search the document corpus and return the most relevant chunks.

    Args:
        query: The natural-language search query.
        k: Number of results to return (default 5).
        context: Number of neighbouring chunks to include before/after each match.
        collection: ChromaDB collection name. Default: stella_contextual.
    """
    col = collection or COLLECTION
    vectorstore = _get_vectorstore(collection_name=col)
    results = vectorstore.similarity_search_with_score(query, k=k)

    output = []
    for doc, score in results:
        # Escape content to prevent JSON serialization issues
        import re
        def escape_content(text):
            # Remove null bytes and other problematic characters
            text = text.replace('\x00', '')
            # Escape control characters
            text = re.sub(r'[\x01-\x1f\x7f]', lambda m: f'\\u{ord(m.group(0)):04x}', text)
            return text
        
        entry = {
            "content": escape_content(doc.page_content),
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "language": doc.metadata.get("language", "unknown"),
        }

        if context > 0:
            source_file = doc.metadata.get("source_file", "unknown")
            chunk_index = doc.metadata.get("chunk_index", -1)
            if chunk_index >= 0:
                entry["context_chunks"] = _get_context_chunks(
                    source_file, chunk_index, context, collection_name=col
                )

        output.append(entry)

    return json.dumps({"results": output}, ensure_ascii=False, indent=2)


@mcp.tool()
def rag_list_sources(collection: Optional[str] = None) -> str:
    """List all unique source files in the document corpus.

    Args:
        collection: Collection name (default: stella_contextual).
    """
    col_name = collection or COLLECTION
    host, port = _parse_chroma_url(CHROMA_URL)
    client = chromadb.HttpClient(host=host, port=port)
    col = client.get_or_create_collection(name=col_name)

    results = col.get(include=["metadatas"])
    sources = sorted(
        set(
            m.get("source_file", m.get("source", "unknown"))
            for m in results.get("metadatas", [])
        )
    )

    return json.dumps(
        {"collection": col_name, "sources": sources}, ensure_ascii=False, indent=2
    )


@mcp.tool()
def rag_list_collections() -> str:
    """List all available collections with their source counts."""
    host, port = _parse_chroma_url(CHROMA_URL)
    client = chromadb.HttpClient(host=host, port=port)
    cols = client.list_collections()

    info = []
    for c in cols:
        col = client.get_collection(c.name)
        count = col.count()
        try:
            results = col.get(include=["metadatas"], limit=1)
            metas = results.get("metadatas", [])
            source_key = (
                "source_file" if metas and "source_file" in metas[0] else "source"
            )
            all_results = col.get(include=["metadatas"])
            sources = sorted(
                set(
                    m.get(source_key, "unknown")
                    for m in all_results.get("metadatas", [])
                    if m
                )
            )
            info.append(
                {
                    "name": c.name,
                    "chunks": count,
                    "source_files": len(sources),
                    "files": sources[:10] + (["..."] if len(sources) > 10 else []),
                }
            )
        except Exception:
            info.append({"name": c.name, "chunks": count})

    return json.dumps(
        {"collections": info, "default": COLLECTION},
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def graph_query(cypher: str, limit: int = 50) -> str:
    """Execute a Cypher query against the Neo4j knowledge graph.

    Args:
        cypher: Cypher query string
        limit: Maximum number of records to return (safety cap, default 50)
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(cypher).data()
            return json.dumps({"results": result[:limit]}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        driver.close()


if __name__ == "__main__":
    mcp.run()
