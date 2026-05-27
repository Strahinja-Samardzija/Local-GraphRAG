#!/usr/bin/env python3
"""
Ingests Japanese, Korean, and English documents using Qwen3-Embedding-8B.
Supports incremental updates: only new or modified files are re-indexed.

Multi-collection mode: subfolders under DOCS_DIR map to separate ChromaDB collections.
  docs/qwen3/   → collection "qwen3"
  docs/jinaai/  → collection "jinaai"

Files placed directly in docs/ (no subfolder) go into the default "documents" collection.
"""

import json
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from qwen3_embeddings import Qwen3Embeddings
from qwen3_embeddings_server import Qwen3EmbeddingsServer
from jina_embeddings import JinaEmbeddings
from stella_embeddings import StellaEmbeddings

DOCS_DIR = "./docs"
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
DEFAULT_COLLECTION = "documents"

# Per-collection chunking overrides
COLLECTION_CHUNK_CONFIG = {
}


def load_documents(directory):
    """Load documents from the specified directory (flat, no subfolders).
    Returns list of (file_path, document) tuples."""
    documents = []
    for file_path in sorted(Path(directory).glob("*")):
        if file_path.is_dir():
            continue
        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".md":
            # Use TextLoader for md files - UnstructuredMarkdownLoader
            # has bugs with large/complex files and silently truncates
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            continue
        docs = loader.load()
        documents.extend([(file_path.name, doc) for doc in docs])
    return documents


def discover_collections(docs_dir):
    """Discover collections from subfolders under docs_dir.
    Returns list of (collection_name, subfolder_path) tuples.
    Files directly in docs_dir go into DEFAULT_COLLECTION."""
    collections = []
    docs_path = Path(docs_dir)

    # Check for loose files in root docs dir
    loose_files = [f for f in docs_path.iterdir() if f.is_file()]
    if loose_files:
        collections.append((DEFAULT_COLLECTION, docs_path))

    # Check subfolders
    for subdir in sorted(docs_path.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            collections.append((subdir.name, subdir))

    return collections


def chunk_documents(documents, chunk_size=None, chunk_overlap=None):
    """Split documents into chunks with multilingual separator support."""
    cs = chunk_size or CHUNK_SIZE
    co = chunk_overlap or CHUNK_OVERLAP
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=cs,
        chunk_overlap=co,
        separators=[
            "\n\n",
            "\n",
            "。",
            "．",
            "？",
            "！",
            "」",
            "】",
            ")",
            " ",
            ""
        ],
        keep_separator=True
    )
    return text_splitter.split_documents(documents)


def detect_language(text):
    """Detect if text is Japanese, Korean, or English."""
    if any('\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in text):
        return "ja"
    elif any('\uAC00' <= c <= '\uD7AF' for c in text):
        return "ko"
    else:
        return "en"


def get_embedding_for_collection(collection_name):
    """Return the appropriate embedding class for a collection."""
    if collection_name == "jinaai":
        return JinaEmbeddings()
    elif collection_name == "stella":
        return StellaEmbeddings()
    else:
        return Qwen3Embeddings()


def get_manifest_path(collection_name):
    """Get the manifest path for a given collection."""
    return os.path.join(CHROMA_PERSIST_DIR, f"manifest_{collection_name}.json")


def get_file_mtime(path):
    """Get file modification time as integer."""
    return int(os.path.getmtime(path))


def load_manifest(manifest_path):
    """Load the index manifest tracking previously ingested files.
    Returns dict: {filename: mtime}"""
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest_path, manifest):
    """Save the updated manifest."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def identify_changes(docs, manifest, docs_dir=None):
    """Identify new, modified, and removed files.
    
    Args:
        docs: list of (name, doc) tuples
        manifest: dict of {filename: mtime}
        docs_dir: base directory for resolving file paths
    """
    base = docs_dir or DOCS_DIR
    current_files = {}
    for name, _ in docs:
        full_path = os.path.join(base, name)
        if os.path.exists(full_path):
            current_files[name] = get_file_mtime(full_path)
        else:
            current_files[name] = 0  # treat as new if we can't stat

    # Unique filenames (deduplicate multi-page docs)
    doc_files = set()
    for name, _ in docs:
        doc_files.add(name)

    new_files = set()
    modified_files = set()
    unchanged_files = set()

    for name in doc_files:
        if name not in manifest:
            new_files.add(name)
        elif current_files[name] != manifest[name]:
            modified_files.add(name)
        else:
            unchanged_files.add(name)

    removed_files = set(manifest.keys()) - doc_files

    return new_files, modified_files, unchanged_files, removed_files


def delete_file_chunks(vectorstore, source_file):
    """Delete all chunks belonging to a specific source file."""
    collection = vectorstore._collection
    results = collection.get(
        where={"source_file": source_file},
        include=["metadatas"]
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results.get("ids", []))


def ingest_collection(docs_dir, collection_name, embeddings, force=False):
    """Ingest documents from a specific directory into a named collection."""
    manifest_path = get_manifest_path(collection_name)

    # Apply per-collection chunk config
    chunk_cfg = COLLECTION_CHUNK_CONFIG.get(collection_name, {})
    c_size = chunk_cfg.get("chunk_size", CHUNK_SIZE)
    c_overlap = chunk_cfg.get("chunk_overlap", CHUNK_OVERLAP)

    print(f"\n{'='*60}")
    print(f"  Collection: {collection_name}")
    print(f"  Source: {docs_dir}")
    print(f"{'='*60}")

    print("Loading documents...")
    docs = load_documents(docs_dir)

    if not docs:
        print(f"  No documents found in {docs_dir}. Skipping.")
        return

    # Flatten docs for processing
    flat_docs = [(name, doc) for name, doc in docs]

    # Load manifest
    if force:
        print("  [FORCE] Full reindex requested. Starting fresh.")
        manifest = {}
    else:
        manifest = load_manifest(manifest_path)

    new_files, modified_files, unchanged_files, removed_files = identify_changes(
        flat_docs, manifest, docs_dir=docs_dir
    )

    if not force and not new_files and not modified_files and not removed_files:
        print("  All files are up to date. Nothing to do.")
        return

    # Report changes
    print(f"  New files:       {len(new_files)} {list(new_files) if new_files else ''}")
    print(f"  Modified files:  {len(modified_files)} {list(modified_files) if modified_files else ''}")
    print(f"  Unchanged files: {len(unchanged_files)}")
    print(f"  Removed files:   {len(removed_files)} {list(removed_files) if removed_files else ''}")

    # Load existing store or create new one
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # Delete chunks for modified/removed files
    for fname in sorted(modified_files | removed_files):
        deleted = delete_file_chunks(vectorstore, fname)
        if deleted:
            print(f"  Removed {deleted} chunks for '{fname}'")

    # Collect chunks for new/modified files
    chunks_to_add = []
    for name, doc in flat_docs:
        if name in new_files or name in modified_files:
            doc.metadata["source_file"] = name
            chunks_to_add.append(doc)

    if chunks_to_add:
        print(f"\n  Chunking {len(chunks_to_add)} document segments... (size={c_size}, overlap={c_overlap})")
        chunks = chunk_documents(chunks_to_add, chunk_size=c_size, chunk_overlap=c_overlap)
        print(f"  Created {len(chunks)} chunks")

        print("  Applying metadata...")
        file_chunk_counts = {}
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source_file", "unknown")
            file_chunk_counts[source] = file_chunk_counts.get(source, 0)
            chunk.metadata["chunk_index"] = file_chunk_counts[source]
            file_chunk_counts[source] += 1
            chunk.metadata["language"] = detect_language(chunk.page_content)

        print(f"  Adding {len(chunks)} chunks to collection '{collection_name}'...")
        vectorstore.add_documents(chunks)

    # Update manifest
    unique_docs = set(name for name, _ in flat_docs)
    for name in unique_docs:
        full_path = os.path.join(docs_dir, name)
        if os.path.exists(full_path):
            manifest[name] = get_file_mtime(full_path)
    for name in removed_files:
        manifest.pop(name, None)

    save_manifest(manifest_path, manifest)
    print(f"  Manifest updated: {manifest_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector store with multi-collection support"
    )
    parser.add_argument("--force", action="store_true",
                        help="Force full reindex (ignores existing store)")
    parser.add_argument("--server", action="store_true",
                        help="Use persistent embedding server instead of loading model locally")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080",
                        help="URL of the embedding server (default: http://127.0.0.1:8080)")
    parser.add_argument("--collection", default=None,
                        help="Ingest only a specific collection (subfolder name), e.g. 'qwen3' or 'jinaai'")
    parser.add_argument("--list-collections", action="store_true",
                        help="List discovered collections without ingesting")
    args = parser.parse_args()

    # Discover collections
    collections = discover_collections(DOCS_DIR)

    if args.list_collections:
        print("Discovered collections:")
        for name, path in collections:
            file_count = len(list(Path(path).glob("*.md"))) + \
                         len(list(Path(path).glob("*.txt"))) + \
                         len(list(Path(path).glob("*.pdf")))
            print(f"  {name}: {file_count} files ({path})")
        return

    # Filter to requested collection
    if args.collection:
        collections = [(n, p) for n, p in collections if n == args.collection]
        if not collections:
            print(f"Collection '{args.collection}' not found.")
            print("Available collections:")
            for n, p in discover_collections(DOCS_DIR):
                print(f"  {n}")
            return

    if not collections:
        print(f"No documents found in {DOCS_DIR}. Please add documents first.")
        return

    # Ingest each collection with its appropriate embedding model
    for collection_name, docs_dir in collections:
        if args.server:
            print(f"Using embedding server at {args.server_url}")
            embeddings = Qwen3EmbeddingsServer(base_url=args.server_url)
        else:
            print(f"Auto-selecting embedding for collection '{collection_name}'...")
            embeddings = get_embedding_for_collection(collection_name)
        ingest_collection(docs_dir, collection_name, embeddings, force=args.force)

    print(f"\n{'='*60}")
    print("All collections processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
