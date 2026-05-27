FROM python:3.12-slim

WORKDIR /app

# System deps for building Python packages (torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY stella_embedding_server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code and Stella embeddings module
COPY stella_embedding_server/server.py /app/server.py
COPY stella_embeddings.py /app/

# Persist ONNX model cache
RUN mkdir -p /app/stella_onnx_cache
ENV STELLA_ONNX_CACHE=/app/stella_onnx_cache

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
