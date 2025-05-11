# Course Chat Bot

This is a course specific chat bot that could be applied to any course with provided course material. The bot is implemented by an underlying Retrieval-Augmented Generation (RAG) system that processes PDF documents and answers questions using the GMI API.

## Overview

This project implements a RAG system that:
- Processes PDF documents from the rawdata directory
- Chunks documents and creates embeddings using sentence-transformers
- Stores embeddings in a FAISS vector index for efficient retrieval
- Uses the GMI API with Deepseek models for question answering
- Provides both a REST API and command-line interface

## Features

- 📄 PDF document processing and chunking
- 🔍 Semantic search using FAISS vector database
- 🤖 Question answering powered by GMI API with Deepseek models
- 🌐 REST API for integration with other applications
- 💻 Interactive command-line interface

## Setup

### Prerequisites

- Python 3.10+
- GMI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bosyuan/Course-bot.git
   cd Course-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add a `.env` file and put your GMI API key in the .env file:
   ```
   GMI_API_KEY='your_api_key_here'
   ```

4. Place your course material PDF documents in the rawdata directory

5. Generate the vector index:
   ```bash
   python vector_store.py
   ```

## Usage

### REST API

Start the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Send queries to the API:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Your question here"}'
```

### Command Line Interface

Run the interactive CLI:
```bash
python rag.py
```

Type your questions at the prompt and receive answers based on your documents.

## Docker Deployment

Build and run with Docker:
```bash
docker build -t rag-app .
docker run -p 8000:8000 rag-app
```

## Project Structure

```
RAG/
├── app.py             # FastAPI application
├── model.py           # GMI API LLM wrapper
├── rag_chain.py       # RAG chain implementation
├── vector_store.py    # Document processing and indexing
├── rag.py             # CLI interface
├── rawdata/           # Directory for PDF documents
├── index/             # FAISS vector store location
├── Dockerfile         # Container configuration
├── requirements.txt   # Project dependencies
└── .env               # Environment variables (API keys)
```

## How It Works

1. PDF documents are loaded and split into chunks
2. Text chunks are embedded using sentence-transformers
3. Embeddings are stored in a FAISS vector index
4. When a query is received, similar documents are retrieved
5. Relevant documents and query are sent to the GMI API
6. The model generates an answer based on the context and query