# Medical Chatbot using Retrieval-Augmented Generation (RAG)

An AI-powered medical Q&A chatbot that answers user queries by retrieving relevant context from a medical reference book and generating grounded responses using GPT-4o. Built with LangChain, Pinecone (vector database), Flask, and HuggingFace sentence embeddings, and containerized with Docker for deployment.

## Overview

Instead of relying purely on an LLM's internal knowledge (which can hallucinate), this project implements a **Retrieval-Augmented Generation (RAG)** pipeline:

1. A medical reference PDF is loaded, chunked, and converted into vector embeddings.
2. The embeddings are stored in a **Pinecone** vector index.
3. When a user asks a question, the most relevant text chunks are retrieved from Pinecone via similarity search.
4. The retrieved context is passed to **GPT-4o** along with the question, so the model answers using grounded, source-backed information instead of guessing.
5. The answer is returned through a simple Flask-based chat UI.

## Features

- **Semantic search over medical literature** — retrieves the top-k most relevant passages for any user question instead of keyword matching.
- **Context-grounded answers** — the LLM is instructed to answer only from retrieved context and admit when it doesn't know.
- **Concise responses** — system prompt caps answers at three sentences to keep the chatbot focused and readable.
- **Persistent vector store** — embeddings are generated once (`store_index.py`) and reused across app runs, so the app doesn't re-embed the PDF on every request.
- **Simple web chat interface** — Flask + HTML/CSS front end for asking questions and viewing responses.
- **Containerized & cloud-deployable** — Dockerfile included, with a documented AWS EC2 + ECR CI/CD deployment flow using GitHub Actions.

## Skills Used

- Python
- Flask (REST endpoint design)
- LangChain (RetrievalQA chain, PromptTemplate)
- OpenAI GPT-4o API integration
- Pinecone (vector index, serverless spec)
- HuggingFace Sentence-Transformers (MiniLM embeddings)
- PyPDF (PDF text extraction)
- RecursiveCharacterTextSplitter (text chunking)
- Docker (containerization)
- Git & GitHub
- AWS EC2
- AWS ECR
- GitHub Actions (CI/CD)
- HTML/CSS (chat UI)
- python-dotenv (env config management)

## How It Works (Architecture)

```
Medical_book.pdf
      │
      ▼
DirectoryLoader + PyPDFLoader  (load_pdf_file)
      │
      ▼
filter_to_minimal_docs()  → keeps only page_content + source metadata
      │
      ▼
RecursiveCharacterTextSplitter  (chunk_size=500, chunk_overlap=20)
      │
      ▼
HuggingFace Embeddings (all-MiniLM-L6-v2, 384 dims)
      │
      ▼
Pinecone Serverless Index ("medical-chatbot", cosine metric)
      │
      ▼
[Runtime] User query ──► Retriever (top-k=3 similarity search)
      │
      ▼
create_stuff_documents_chain + ChatOpenAI (gpt-4o) + system_prompt
      │
      ▼
create_retrieval_chain → Answer
      │
      ▼
Flask /get endpoint → chat.html UI
```

## Project Structure

```
medical-chatbot-llm/
├── app.py                 # Flask app: loads index, builds RAG chain, serves chat UI
├── store_index.py         # One-time script: embeds PDF and upserts vectors to Pinecone
├── src/
│   ├── helper.py           # PDF loading, doc filtering, text splitting, embedding model loader
│   └── prompt.py            # System prompt for the QA chain
├── templates/
│   └── chat.html            # Chat UI template
├── static/
│   └── style.css             # Chat UI styling
├── data/
│   └── Medical_book.pdf     # Source medical reference document
├── research/
│   └── trials.ipynb         # Experimentation notebook
├── requirements.txt
├── setup.py
├── Dockerfile
└── template.sh              # Bootstraps the project folder/file structure
```

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Prince95086/medical-chatbot-llm.git
cd medical-chatbot-llm
```

### 2. Create a conda environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```ini
PINECONE_API_KEY="your-pinecone-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

### 5. Build the vector index (run once)

This loads `data/Medical_book.pdf`, chunks it, generates embeddings, and stores them in Pinecone.

```bash
python store_index.py
```

### 6. Run the app

```bash
python app.py
```

Open your browser at `http://localhost:8080`.

## Docker

Build and run the app in a container:

```bash
docker build -t medical-chatbot .
docker run -p 8080:8080 --env-file .env medical-chatbot
```

## Deployment (AWS EC2 + ECR via GitHub Actions)

The repo documents a CI/CD flow for deploying to AWS:

1. Build a Docker image of the app.
2. Push the image to **Amazon ECR** (Elastic Container Registry).
3. Launch an **EC2** instance and configure it as a self-hosted GitHub Actions runner.
4. On push, the pipeline pulls the image from ECR onto EC2 and runs it.

Required GitHub Actions secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REPO`
- `PINECONE_API_KEY`
- `OPENAI_API_KEY`

Required IAM permissions: `AmazonEC2ContainerRegistryFullAccess`, `AmazonEC2FullAccess`.

## Key Implementation Details

- **Chunking strategy**: `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=20` — small chunks with slight overlap to preserve context continuity across boundaries.
- **Retriever config**: similarity search with `k=3`, so each answer is generated from the 3 most relevant chunks.
- **Guardrails**: the system prompt explicitly instructs the model to say "I don't know" when the retrieved context doesn't answer the question, reducing hallucination risk.
- **Index reuse**: `app.py` connects to an *existing* Pinecone index (`from_existing_index`) rather than rebuilding it, keeping the app lightweight at request time.

## License

This project is for educational and demonstration purposes.

