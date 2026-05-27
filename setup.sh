#!/bin/bash
# Setup script for SOTA Local RAG System
# This script sets up the virtual environment and installs dependencies

set -e

echo "=== SOTA Local RAG System Setup ==="

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt-get install -y python3 python3-pip python3-venv build-essential

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p docs
mkdir -p chroma_db
mkdir -p models

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Place your documents in the ./docs/ folder"
echo "3. Start required services (see instructions below)"
echo "4. Run document ingestion: python ingest_documents.py"
echo "5. Start MCP server: mcp run run_mcp_server.py"
echo ""
echo "=== SERVICE STARTUP INSTRUCTIONS ==="
echo ""
echo "Option A: Start all services with Docker Compose"
echo "  docker-compose up -d neo4j"
echo "  python start_services.py --stella"
echo ""
echo "Option B: Start services manually"
echo "  # Neo4j (via Docker)"
echo "  docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:5.15.0"
echo ""
echo "  # ChromaDB"
echo "  chroma run --host 0.0.0.0 --port 8000 --path ./chroma_db"
echo ""
echo "  # Stella Embedding Server (ONNX - auto-downloads on first run)"
echo "  python stella_embedding_server/server.py"
echo ""
echo "  # Qwen3 Embedding Server (GGUF - requires manual model download)"
echo "  # Download: huggingface-cli download Qwen/Qwen3-Embedding-8B-GGUF Qwen3-Embedding-8B-f16.gguf --local-dir ./models"
echo "  python -m llama_cpp.server --model ./models/Qwen3-Embedding-8B-f16.gguf --host 127.0.0.1 --port 8080 --embedding true"
echo ""
echo "=== MODEL SETUP ==="
echo ""
echo "Stella (ONNX): Auto-downloads on first request (~1.5GB)"
echo "  - No manual download needed"
echo "  - First run will convert and cache to ./stella_onnx_cache/"
echo ""
echo "Qwen3 (GGUF): Manual download required (~5.5GB)"
echo "  huggingface-cli download Qwen/Qwen3-Embedding-8B-GGUF Qwen3-Embedding-8B-f16.gguf --local-dir ./models"
echo ""
echo "Jina (GGUF): Manual download required (~0.5GB)"
echo "  # Download jina-embeddings-v2-base-code-f16.gguf to ./models/"
echo ""
echo "=== MCP SERVER CONFIGURATION ==="
echo ""
echo "Configure via environment variables:"
echo "  CHROMA_URL=http://localhost:8000"
echo "  EMBEDDING_URL=http://localhost:8080"
echo "  COLLECTION=stella_contextual"
echo "  STELLA_EMBEDDING_URL=http://localhost:8081"
echo "  NEO4J_URI=bolt://localhost:7687"
echo "  NEO4J_USER=neo4j"
echo "  NEO4J_PASSWORD=your_password"
