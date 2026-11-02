# Enterprise AI Knowledge & Data Agent

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents, search their contents semantically, and ask questions with AI-generated answers grounded in retrieved information.

**Live Demo:** [Open Application](https://enterprise-ai-agent-lime.vercel.app/)

**Backend API:** [Render Backend](https://enterprise-ai-agent-ohvl.onrender.com/)

**API Documentation:** [Swagger UI](https://enterprise-ai-agent-ohvl.onrender.com/docs)

---

## Overview

Enterprise AI Knowledge & Data Agent is a document-grounded AI application built using React, FastAPI, PostgreSQL, Qdrant, Cloudflare Workers AI, and Groq.

Users can upload documents, which are processed into text chunks and embeddings. When a user asks a question, the application retrieves relevant document chunks and passes the retrieved context to an LLM to generate a grounded response with source citations.

The application includes authentication, user-scoped document access, semantic search, conversation history, and safe responses when relevant information is unavailable.

## Key Features

- **Document ingestion:** Upload PDF, DOCX, TXT, and Markdown documents.
- **Text extraction:** Extract readable content from supported document formats.
- **Text chunking:** Divide extracted content into smaller passages for retrieval.
- **Embedding generation:** Generate embeddings using Cloudflare Workers AI and BGE Small English.
- **Vector search:** Store and retrieve document embeddings using Qdrant.
- **RAG question answering:** Generate context-grounded responses using Groq.
- **Source citations:** Display source documents, chunk information, and retrieval relevance.
- **Safe abstention:** Return a fallback response when information cannot be found in the provided documents.
- **Authentication:** User registration, login, and protected APIs using JWT.
- **User-scoped access:** Ownership checks for document and conversation operations.
- **Conversation history:** Maintain conversations and support follow-up questions.
- **Document management:** Upload, list, and delete documents.
- **Automated testing:** 54 backend tests passed in the latest verified run.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite, JavaScript |
| Backend | Python, FastAPI |
| Relational database | Neon PostgreSQL |
| Vector database | Qdrant Cloud |
| Embeddings | Cloudflare Workers AI – BGE Small English |
| LLM inference | Groq API |
| LLM model | `openai/gpt-oss-20b` |
| Authentication | JWT |
| Document processing | pypdf, python-docx |
| Testing | pytest |
| Containerization | Docker, Docker Compose |
| Frontend deployment | Vercel |
| Backend deployment | Render |

## System Architecture

```text
                         USER
                          |
                          v
                 React + Vite Frontend
                       (Vercel)
                          |
                          v
                    FastAPI Backend
                       (Render)
                          |
                   Document Ingestion
                          |
                   Text Extraction
                          |
                      Chunking
                          |
                          v
               Cloudflare Workers AI
                   BGE Embeddings
                          |
                          v
                     Qdrant Cloud
                   Vector Storage
                          |
                          v
                  Semantic Retrieval
                          |
                          v
                      Groq API
                 LLM Answer Generation
                          |
                          v
                Grounded Answer + Citations
                          |
                          v
                  React Chat Interface


                   Neon PostgreSQL
                   - User accounts
                   - Document metadata
                   - Extracted chunks
                   - Conversation data
```

## RAG Workflow

1. A user authenticates and uploads a supported document.
2. FastAPI validates the uploaded file.
3. The backend extracts readable text.
4. Extracted text is divided into smaller chunks.
5. Cloudflare Workers AI generates embeddings.
6. Document metadata and extracted chunks are stored in PostgreSQL.
7. Embeddings and associated metadata are stored in Qdrant Cloud.
8. When a user asks a question, the backend generates a query embedding.
9. Qdrant retrieves relevant chunks using semantic similarity.
10. Retrieved context is passed to Groq for answer generation.
11. The application returns the answer with source citations.

## Production Deployment

| Service | Platform |
|---|---|
| Frontend | Vercel |
| Backend | Render |
| PostgreSQL | Neon |
| Vector database | Qdrant Cloud |
| Embeddings | Cloudflare Workers AI |
| LLM inference | Groq |

### Production URLs

- [Frontend Application](https://enterprise-ai-agent-lime.vercel.app/)
- [Backend Health Check](https://enterprise-ai-agent-ohvl.onrender.com/health)
- [Swagger API Documentation](https://enterprise-ai-agent-ohvl.onrender.com/docs)

## Local Development Setup

### Prerequisites

- Python 3.10+
- Node.js and npm
- Git
- Docker Desktop (for local PostgreSQL and Qdrant)

### 1. Clone the repository

```bash
git clone https://github.com/sandeepkumarpittala/enterprise-ai-agent.git
cd enterprise-ai-agent
```

### 2. Configure environment variables

Configure the backend environment using the required variable names from your project configuration.

Required services include:

- PostgreSQL
- JWT authentication
- Qdrant
- Groq API
- Cloudflare Workers AI

Never commit `.env` files or expose credentials.

### 3. Set up the backend

```powershell
cd backend

python -m venv .venv

..venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Start the backend from the project root:

```powershell
python -m uvicorn app.main:app --app-dir backend --reload
```

Health check: `http://127.0.0.1:8000/health`

API documentation: `http://127.0.0.1:8000/docs`

### 4. Set up the frontend

Open another terminal:

```powershell
cd frontend

npm install

npm run dev
```

Configure `VITE_API_BASE_URL` to point to your backend URL.

### 5. Run backend tests

From the backend directory:

```powershell
pytest -q
```

Latest verified result:

```text
54 passed
```

## Project Structure

```text
enterprise-ai-agent/
|
|-- backend/
|   |-- app/
|   |   |-- agents/
|   |   |-- api/
|   |   |-- core/
|   |   |-- db/
|   |   |-- evaluation/
|   |   |-- models/
|   |   |-- rag/
|   |   |-- schemas/
|   |   |-- services/
|   |   |-- tools/
|   |   |-- main.py
|   |
|   |-- tests/
|   |-- requirements.txt
|
|-- frontend/
|   |-- src/
|   |-- public/
|   |-- package.json
|   |-- vite.config.js
|
|-- docker-compose.yml
|-- README.md
```

*Directory listings are representative of the main application structure.*

## Testing and Verification

The following checks have been completed:

- Backend automated tests: 54 passed.
- Frontend ESLint: passed.
- Frontend production build: passed.
- Production frontend and backend connectivity: verified.
- Document upload and processing: verified.
- PostgreSQL document and chunk retrieval: verified.
- Semantic retrieval: verified.
- RAG answer generation: verified.
- Source citations: verified.
- Unanswerable question fallback: verified.

These checks demonstrate the core application workflow, not large-scale production performance.

## Future Improvements

- Larger RAG evaluation benchmark.
- Retrieval quality measurement and optimization.
- Hybrid search and reranking.
- More comprehensive multi-document testing.
- Database and vector database backup and recovery.
- Improved monitoring and error handling.
- Persistent original-file storage and downloads, if required.
- Advanced agent and tool-execution capabilities.

## Author

**Pittala Sandeep Kumar**

- GitHub: [sandeepkumarpittala](https://github.com/sandeepkumarpittala)
- LinkedIn: [Sandeep Kumar](https://www.linkedin.com/in/sandeep-kumar-pittala-06025b363/)