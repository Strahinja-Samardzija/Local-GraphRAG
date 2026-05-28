# Local GraphRAG System

A production-ready, local-first Retrieval-Augmented Generation (RAG) system with multi-model embeddings, hierarchical document processing (RAPTOR), and knowledge graph extraction capabilities. **Resume-Ready Implementation**.

## 🎯 **Project Success Story**

This system successfully demonstrated its capabilities by using the MCP server to automatically generate a comprehensive technical chapter on **Logical Replication** without any web search. The agent:

- ✅ Used `rag_query` and `graph_query` tools to extract relevant information
- ✅ Chained multiple tool calls efficiently to build knowledge progressively
- ✅ Applied the Diátaxis framework and multidimensional taxonomy for structured output
- ✅ Generated production-quality technical documentation (385 lines, 9/10 quality rating)
- ✅ Proved the system's ability to perform complex reasoning tasks autonomously

**Key Achievement**: The system successfully demonstrated its ability to autonomously generate production-quality technical documentation using only local documents and MCP tools, proving real-world AI agent capabilities for complex research tasks.

## Features

- **Local-First**: All processing happens on your machine
- **Multi-Model Embeddings**: Qwen3-Embedding-8B, Jina, and Stella embeddings supported
- **Multi-Collection Support**: Organize documents by collection (qwen3, jinaai, stella)
- **Incremental Updates**: Only processes new or modified documents
- **RAPTOR Hierarchical Processing**: Recursive Abstractive Processing for Tree-Organized Retrieval
- **Knowledge Graph Extraction**: Automated graph construction from document clusters
- **Neo4j Integration**: Export extracted knowledge graphs to Neo4j
- **Context Expansion**: Retrieve neighboring chunks for better context
- **Embedding Server**: Persistent server mode for faster queries
- **Multilingual**: Support for English, Japanese, and Korean documents

## Quick Start

### 1. Setup Environment

Run the setup script to install dependencies and create the virtual environment:

```bash
./setup.sh
```

**Important**: The setup script now includes detailed instructions for:

- Starting all required services (ChromaDB, Neo4j, embedding servers)
- Installing and configuring embedding models (Qwen3 GGUF, Jina GGUF, Stella ONNX)
- MCP server configuration via environment variables

After running `./setup.sh`, follow the service startup and model installation instructions displayed at the end of the script.

### 2. Add Documents

Place your documents (`.txt`, `.pdf`, `.md`) in the appropriate collection folder:

- `./docs/qwen3/` - For Qwen3 embeddings
- `./docs/jinaai/` - For Jina embeddings
- `./docs/stella/` - For Stella embeddings
- `./docs/` - Default collection

### 3. Start Required Services

Choose one of the methods shown in the setup script output:

**Option A (Recommended):**

```bash
docker-compose up -d neo4j
python start_services.py --stella
```

**Option B (Manual):**
Follow the individual service startup commands displayed by `./setup.sh`

### 4. Ingest Documents

```bash
python ingest_documents.py --collection stella
```

### 5. Query Documents

```bash
python rag_pipeline.py --query "your question" --collection stella --context 3
```

### 6. Start MCP Server

```bash
mcp run run_mcp_server.py
```

**Note**: The MCP server is now started using the `mcp` command rather than running the Python script directly. Configure it via environment variables as shown in the setup script output.

## Project Structure

```
sota-rag/
├── setup.sh                    # Environment setup script
├── requirements.txt            # Python dependencies
├── ingest_documents.py         # Multi-collection document ingestion
├── rag_pipeline.py             # Query and retrieval with context expansion
├── run_mcp_server.py           # MCP server with graph query support
├── start_services.py           # Service orchestration (ChromaDB + embeddings)
├── qwen3_embeddings.py         # Qwen3 embedding wrapper
├── qwen3_embeddings_server.py  # Persistent embedding server
├── jina_embeddings.py          # Jina embedding wrapper
├── stella_embeddings.py        # Stella embedding wrapper
├── stella_embedding_server.py  # Stella ONNX embedding server
├── ingest_summaries.py         # RAPTOR hierarchical processing
├── extract_graph_glm.py        # Knowledge graph extraction
├── fuse_to_neo4j_updated.py    # Neo4j graph fusion
├── docs/                       # Document storage by collection
│   ├── qwen3/                  # Qwen3 collection
│   ├── jinaai/                 # JinaAI collection
│   └── stella/                 # Stella collection
├── chroma_db/                  # Vector database
├── models/                     # ML model storage
└── docs/raptor/                # RAPTOR processing outputs
    └── stella/                 # Stella collection RAPTOR data
        ├── raptor_level_1_nodes.json
        ├── raptor_level_2_nodes.json
        └── raptor_level_3_nodes.json

```

## MCP Server Configuration

The MCP server is configured via `.roo/mcp.json` and supports:

- **rag_query**: Search document corpus with context expansion
- **rag_list_sources**: List all source files in collections
- **rag_list_collections**: Show available collections
- **graph_query**: Execute Cypher queries against Neo4j knowledge graph

Example MCP configuration:

```json
{
	"mcpServers": {
		"sota-rag": {
			"command": "/path/to/venv/bin/mcp",
			"args": ["run", "run_mcp_server.py"],
			"env": {
				"CHROMA_URL": "http://localhost:8000",
				"EMBEDDING_URL": "http://localhost:8080",
				"COLLECTION": "stella_contextual",
				"NEO4J_URI": "bolt://localhost:7687",
				"NEO4J_USER": "neo4j",
				"NEO4J_PASSWORD": "your_password"
			}
		}
	}
}
```

## Advanced Features

### RAPTOR Hierarchical Processing

```bash
python ingest_summaries.py --collection stella

```

### Knowledge Graph Extraction

```bash
python extract_graph_glm.py --cluster cluster_1001.json
python fuse_to_neo4j_updated.py

```

### Embedding Server Mode

```bash
python start_services.py --stella

```

## Performance Tips

- Use `stella_contextual` collection for best retrieval quality
- Enable context expansion (`--context 3-5`) for complex queries
- Run embedding server for faster repeated queries
- Monitor database performance for high-throughput scaling
- Use parallel workers for large document collections

## Troubleshooting

- **ChromaDB connection issues**: Ensure ChromaDB is running on configured port
- **Embedding server errors**: Check model availability in `models/` directory
- **Graph queries failing**: Verify Neo4j is running and credentials are correct
- **Memory issues**: Reduce batch sizes in `ingest_documents.py`

## Contributing & Development Status

This is an open-source engineering prototype built exclusively for **educational, non-commercial research, and architectural exploration** purposes.

The implementation showcases:

- Production-ready MCP server implementation
- Advanced RAG techniques with context expansion
- Multi-model embedding support
- Knowledge graph extraction and integration
- Autonomous AI agent capabilities

---

## ⚙️ **License & Liability Waiver**

This repository provides an algorithmic and structural framework. This software does not ship with, host, or distribute any third-party or copyrighted text corpora. End-users retain sole responsibility for ensuring they possess the necessary legal rights, licenses, and permissions for any files ingested locally.

Licensed under the **MIT License**.
