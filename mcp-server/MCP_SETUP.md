# SOTA RAG — MCP Client Setup

Set up a lightweight MCP server on any machine (no model files, no local database) that connects to remote hosting servers.

## Prerequisites

- Python 3.10+
- Network access to the hosting machine's ports **8000** (ChromaDB) and **8080** (embeddings)

## 1. Install

```bash
# Option A: run setup script
bash setup_mcp.sh

# Option B: manual
python3 -m venv venv-mcp
source venv-mcp/bin/activate
pip install -r requirements_mcp.txt
```

## 2. Run

```bash
source venv-mcp/bin/activate

sota_rag_mcp \
  --chroma-url http://<HOST_IP>:8000 \
  --embedding-url http://<HOST_IP>:8080
```

Replace `<HOST_IP>` with the IP of the machine running the hosting servers.

### SSE transport (for network clients)

```bash
sota_rag_mcp \
  --chroma-url http://<HOST_IP>:8000 \
  --embedding-url http://<HOST_IP>:8080 \
  --transport sse --port 8090 --host 0.0.0.0
```

Clients connect to `http://<THIS_MACHINE_IP>:8090/sse`.

## 3. Integrate with Qwen Code

Add to your Qwen Code settings (`~/.qwen/settings.json` or project `.qwen/settings.json`):

```json
{
  "mcpServers": {
    "sota-rag": {
      "command": "/absolute/path/to/venv-mcp/bin/sota_rag_mcp",
      "args": [
        "--chroma-url", "http://<HOST_IP>:8000",
        "--embedding-url", "http://<HOST_IP>:8080"
      ]
    }
  }
}
```

After restarting Qwen Code, the agent will see three tools:
- **`rag_query(query, k=5, context=0, collection="qwen3")`** — search the corpus
- **`rag_list_sources(collection="qwen3")`** — list source files in a collection
- **`rag_list_collections()`** — list all collections with metadata

## 4. Collections

The system maintains separate ChromaDB collections:

| Collection | Embedding Model | Content |
|-----------|-----------------|---------|
| `qwen3` (default) | Qwen3-Embedding-8B | Multilingual texts (JA/KO/EN), books, literature |
| `jinaai` | jina-embeddings-v2-code | English technical docs, API references, code docs |

Use the `collection` parameter in `rag_query` to target the right corpus:

```
# Query wrestling book (multilingual)
rag_query("freestyle wrestling grip techniques", collection="qwen3")

# Query React Native docs (English code)
rag_query("how to use FlatList", collection="jinaai")
```

## 5. Verify

```bash
source venv-mcp/bin/activate

# List collections
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
for c in client.list_collections():
    print(f'{c.name}: {client.get_collection(c.name).count()} chunks')
"

# Test the underlying pipeline
python3 rag_pipeline.py \
  --query "freestyle wrestling techniques" \
  --json --server \
  --chroma-url http://<HOST_IP>:8000 \
  --server-url http://<HOST_IP>:8080
```

## Architecture

```
┌─────────────────────────────────┐
│  Host machine                   │
│  ┌───────────────────────────┐  │
│  │ llama-cpp embedding srv  │  │  :8080
│  │ (Qwen3-Embedding-8B)     │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │ ChromaDB vector store    │  │  :8000
│  │  - qwen3 collection      │  │
│  │  - jinaai collection     │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
               ↕ HTTP
┌─────────────────────────────────┐
│  MCP client machine             │
│  ┌───────────────────────────┐  │
│  │ sota_rag_mcp (stdio/sse) │  │  → Qwen Code
│  │  tools: rag_query        │  │
│  │         rag_list_sources  │  │
│  │         rag_list_colls    │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```
