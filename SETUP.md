# Setup Guide

This document provides complete setup and deployment instructions for the Receipt Intelligence System.

## Quick Start (How to Run)

### Option 1: Docker (Recommended - Easiest)

```bash
# 1. Create .env file with your API keys
copy .env.example .env
# Edit .env and add: OPENAI_API_KEY and PINECONE_API_KEY

# 2. Build and run
docker-compose up --build -d

# 3. Open browser to: http://localhost:8501
```

### Option 2: Manual (Local Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables (or create .env file)
set OPENAI_API_KEY=your-key
set PINECONE_API_KEY=your-key

# 3. Run the application
python run.py

# 4. Open browser to: http://localhost:8501
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Docker Deployment (Recommended)](#docker-deployment-recommended)
4. [Manual Deployment](#manual-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Checklist](#production-checklist)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required API Keys

| Service | Purpose | Get Key At |
|---------|---------|------------|
| OpenAI | Embeddings + LLM responses | https://platform.openai.com/api-keys |
| Pinecone | Vector database | https://app.pinecone.io |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB |
| Storage | 10 GB | 50 GB SSD |
| Network | 10 Mbps | 100 Mbps |

---

## Environment Configuration

### 1. Create Environment File

Create `.env` in project root:

```bash
# Required: API Keys
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here

# Required: Pinecone Configuration
PINECONE_INDEX_NAME=receipt-index

# Optional: Configuration
EMBEDDING_MODEL=text-embedding-3-small
RECEIPT_DATA_PATH=data/receipt_samples_100
STREAMLIT_SERVER_PORT=8501

# Optional: Reference date for temporal queries (testing)
# RECEIPT_REFERENCE_DATE=2024-01-01
```

### 2. Verify Python Installation

```bash
python --version  # Requires 3.9+
pip --version
```

---

## Docker Deployment (Recommended)

Docker provides the simplest, most reproducible deployment method.

### Quick Start

```bash
# 1. Clone/navigate to project
cd receipt-intelligence-system

# 2. Create .env file (see above)
nano .env

# 3. Build and start containers
docker-compose up --build -d

# 4. Access the application
# Open browser: http://localhost:8501
```

### Docker Services Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                     │
│                                                             │
│  ┌──────────────┐        ┌──────────────┐                   │
│  │   receipt-   │        │   receipt-   │                   │
│  │     app      │◄──────►│   indexer    │                   │
│  │  (Streamlit) │        │  (One-time)  │                   │
│  │   Port 8501  │        │              │                   │
│  └──────────────┘        └──────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐        ┌──────────────┐                   │
│  │   Shared     │        │   Pinecone   │                   │
│  │   Volume     │        │   (External) │                   │
│  │  /app/data   │        │              │                   │
│  └──────────────┘        └──────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Building the Docker Image

```bash
# Build image
docker build -t receipt-intelligence:latest .

# Run container
docker run -d \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  --name receipt-app \
  receipt-intelligence:latest
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build -d
```

### Docker Volume Management

```bash
# Persist data between restarts
docker volume create receipt-data

# Backup indexed data
docker run --rm \
  -v receipt-data:/backup \
  -v $(pwd):/output \
  alpine tar czf /output/receipt-backup.tar.gz /backup
```

---

## Manual Deployment

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example environment
cp .env.example .env

# Edit with your API keys
nano .env
```

### Step 3: Index Receipt Data

```bash
# Run indexing script
python -c "
import sys
sys.path.insert(0, '.')
from src.vectorstore.vector_manager import VectorManager
from src.parsers.receipt_parser import ReceiptParser
from src.chunking.receipt_chunker import ReceiptChunker
from src.utils.data_loader import load_receipt_files

# Initialize components
vector_manager = VectorManager()
parser = ReceiptParser()
chunker = ReceiptChunker()

# Clear existing index
vector_manager.clear_index()

# Load and index receipts
receipts = load_receipt_files('data/receipt_samples_100')
all_chunks = []
for receipt_data in receipts:
    receipt = parser.parse(receipt_data['content'], receipt_data['filename'])
    chunks = chunker.chunk_receipt(receipt)
    all_chunks.extend(chunks)

# Index all chunks
vector_manager.index_chunks(all_chunks)
print(f'Indexed {len(all_chunks)} chunks from {len(receipts)} receipts')
"
```

### Step 4: Start Application

```bash
# Run Streamlit UI
python -m streamlit run src/ui/streamlit_app.py

# Or use the launcher
python run.py
```

---

## Cloud Deployment

### AWS Deployment

#### ECS (Elastic Container Service)

```bash
# 1. Push image to ECR
docker build -t receipt-intelligence .
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
docker tag receipt-intelligence:latest $ECR_URL/receipt-intelligence:latest
docker push $ECR_URL/receipt-intelligence:latest

# 2. Create ECS cluster
aws ecs create-cluster --cluster-name receipt-cluster

# 3. Create task definition (receipt-task.json)
# See: ecs-task-definition.json reference

# 4. Run service
aws ecs create-service \
  --cluster receipt-cluster \
  --service-name receipt-service \
  --task-definition receipt-task \
  --desired-count 1
```

#### EC2 Instance

```bash
# User data script for EC2
#!/bin/bash
yum update -y
yum install -y docker
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone and run
cd /opt
git clone https://github.com/your-org/receipt-intelligence.git
cd receipt-intelligence

# Create env file
cat > .env << EOF
OPENAI_API_KEY=${OPENAI_API_KEY}
PINECONE_API_KEY=${PINECONE_API_KEY}
PINECONE_INDEX_NAME=receipt-index
EOF

# Start
docker-compose up -d
```

### Google Cloud Platform

#### Cloud Run

```bash
# Build and push to GCR
docker build -t gcr.io/PROJECT_ID/receipt-intelligence .
docker push gcr.io/PROJECT_ID/receipt-intelligence

# Deploy to Cloud Run
gcloud run deploy receipt-intelligence \
  --image gcr.io/PROJECT_ID/receipt-intelligence \
  --platform managed \
  --port 8501 \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY,PINECONE_API_KEY=$PINECONE_API_KEY
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name receipt-rg --location eastus

# Create container
az container create \
  --resource-group receipt-rg \
  --name receipt-intelligence \
  --image receipt-intelligence:latest \
  --ports 8501 \
  --environment-variables \
    OPENAI_API_KEY=$OPENAI_API_KEY \
    PINECONE_API_KEY=$PINECONE_API_KEY
```

---

## Production Checklist

### Security

- [ ] API keys stored in secrets manager (not in code)
- [ ] `.env` file in `.gitignore`
- [ ] Container runs as non-root user
- [ ] No sensitive data in Docker layers
- [ ] HTTPS enabled (reverse proxy)

### Performance

- [ ] Pinecone index warmed up
- [ ] Receipt data pre-indexed
- [ ] Container resource limits set
- [ ] Health checks configured

### Monitoring

- [ ] Application logs aggregated
- [ ] Error alerting configured
- [ ] Performance metrics tracked
- [ ] Index stats monitoring

### Backup & Recovery

- [ ] Pinecone index backup strategy
- [ ] Receipt data backup
- [ ] Recovery runbook documented

---

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs receipt-app

# Verify env vars
docker exec receipt-app env | grep -i key

# Check resource usage
docker stats receipt-app
```

#### No Results for Queries

1. **Check index stats**:
```python
from src.vectorstore.vector_manager import VectorManager
vm = VectorManager()
print(vm.get_index_stats())
```

2. **Verify data indexed**:
```bash
# Should show > 0 vectors
curl -H "Api-Key: $PINECONE_API_KEY" \
  https://receipt-index.svc.environment.pinecone.io/describe_index_stats
```

3. **Re-index if needed**:
```bash
docker-compose exec app python -c "
from src.vectorstore.vector_manager import VectorManager
vm = VectorManager()
vm.clear_index()
# ... re-index script
"
```

#### Slow Query Performance

- Check Pinecone region (should be close to app)
- Verify OpenAI API latency
- Monitor embedding cache hit rate
- Consider increasing `top_k` for better recall

#### Out of Memory

```bash
# Increase container memory
docker run -m 4g receipt-intelligence:latest

# Or in docker-compose.yml:
services:
  app:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Debug Commands

```bash
# Shell into container
docker exec -it receipt-app /bin/bash

# Check Python environment
docker exec receipt-app python --version
docker exec receipt-app pip list

# Test imports
docker exec receipt-app python -c "from src.query.query_engine import QueryEngine; print('OK')"

# View resource usage
docker system df
docker container ls --size
```

---

## Docker Files Reference

### Files Created

| File | Purpose |
|------|---------|
| `Dockerfile` | Application container definition |
| `docker-compose.yml` | Multi-service orchestration |
| `.dockerignore` | Exclude files from build context |

### Build Arguments

```dockerfile
# Build with custom Python version
docker build --build-arg PYTHON_VERSION=3.11 .
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API access |
| `PINECONE_API_KEY` | Yes | - | Pinecone DB access |
| `PINECONE_INDEX_NAME` | No | `receipt-index` | Vector index name |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `STREAMLIT_SERVER_PORT` | No | `8501` | UI port |
| `RECEIPT_DATA_PATH` | No | `data/receipt_samples_100` | Data directory |

---
