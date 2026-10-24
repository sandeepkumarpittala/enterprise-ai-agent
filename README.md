# Enterprise AI Knowledge & Data Agent



A Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from uploaded documents.



## Overview



This project demonstrates a document-grounded AI application using a Python backend, React frontend, vector search, and a local large language model.



It focuses on document ingestion, semantic retrieval, grounded question answering, conversation follow-ups, and user-scoped data access.



## Key Features



- **Document ingestion:** Upload and process documents for retrieval.

- **RAG pipeline:** Retrieve relevant document chunks to provide context for answering questions.

- **Semantic search:** Generate embeddings and retrieve relevant content using Qdrant.

- **Grounded answers:** Generate responses based on retrieved document context.

- **Safe abstention:** Return a fallback response when the answer cannot be found in the provided documents.

- **Conversation follow-ups:** Use conversation history to resolve follow-up questions.

- **User-scoped access:** Apply ownership checks to document and conversation operations.

- **Automated tests:** Test RAG behavior, evaluation logic, input validation, and access-control paths.

- **RAG evaluation:** Evaluate a dataset containing answerable and unanswerable questions.



## Technology Stack



| Component | Technology |

|---|---|

| Frontend | React, Vite |

| Backend | Python, FastAPI |

| Relational database | PostgreSQL |

| Vector database | Qdrant |

| Local language model | Ollama, `llama3.2:3b` |

| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |

| Testing | pytest |

| Containers | Docker, Docker Compose |



## Architecture



1\. The React frontend provides the user interface.

2\. FastAPI handles application API requests.

3\. Documents are processed into chunks and converted into embeddings.

4\. Qdrant retrieves relevant chunks for a user's question.

5\. The retrieved context and relevant conversation history are used to generate an answer with Ollama.

6\. PostgreSQL stores relational application data, while Qdrant stores vectors and associated payloads.

7\. The evaluation runner checks system behavior against a test dataset.



## Prerequisites



- Python 3.10

- Node.js and npm

- Docker Desktop with Docker Compose

- Ollama



## Local Setup



### 1. Configure environment variables



From the project root, create your local environment file:



```powershell

Copy-Item .env.example .env

```



Fill in the required values for your local setup. Never commit your actual `.env` file or expose credentials.



### 2. Start PostgreSQL and Qdrant



From the project root:



```powershell

docker compose up -d

docker compose ps

```



Confirm both services are running before starting the backend.



### 3. Install backend dependencies



From the project root:



```powershell

cd backend

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

cd ..

```



### 4. Start Ollama



Ensure Ollama is installed and running. Download the configured model if needed:



```powershell

ollama pull llama3.2:3b

```



### 5. Start the backend



From the project root, with the backend virtual environment activated:



```powershell

python -m uvicorn app.main:app --app-dir backend --reload --log-level info

```



Useful endpoints:



- Health check: `http://127.0.0.1:8000/health`

- API documentation: `http://127.0.0.1:8000/docs`



### 6. Start the frontend



Open a second PowerShell terminal:



```powershell

cd frontend

npm install

npm run dev

```



Open the local URL printed by Vite.



## Running Tests



From the project root, with the backend virtual environment activated:



```powershell

python -m pytest backend/tests/ -v

```



The latest verified test run completed with 25 passing tests.



## RAG Evaluation



The evaluation dataset contains 15 answerable and unanswerable question cases.



Run the evaluation from the project root:



```powershell

$env:PYTHONPATH = "$PWD\\backend"

python -m app.evaluation.runner --user-id 5 --limit 5

Remove-Item Env:PYTHONPATH

```



The generated report is written to:



`backend/rag\_evaluation\_report.json`



The latest verified evaluation reported 15/15 passing cases under the project's current automated evaluation logic. This is not a guarantee of perfect answer quality or production readiness.



## Project Structure



```text

enterprise-ai-agent/

â”œâ”€â”€ backend/

â”‚   â”œâ”€â”€ app/

â”‚   â””â”€â”€ tests/

â”œâ”€â”€ frontend/

â”œâ”€â”€ docs/

â”œâ”€â”€ evaluation/

â”œâ”€â”€ .env.example

â”œâ”€â”€ .gitignore

â”œâ”€â”€ alembic.ini

â”œâ”€â”€ docker-compose.yml

â””â”€â”€ README.md

```



## Current Status



- Backend health endpoint verified.

- PostgreSQL and Qdrant configured through Docker Compose.

- RAG question answering and follow-up handling implemented.

- Automated tests: 25 passed in the latest verified run.

- Evaluation dataset: 15 cases passed in the latest verified evaluation.

- Selected cross-user access-control scenarios manually tested.



## Future Improvements



- Expand the evaluation dataset and use stronger semantic evaluation.

- Add retrieval metrics such as precision@k and recall@k.

- Increase integration and security test coverage.

- Improve observability and latency monitoring.

- Complete deployment configuration and production hardening.



## Author



**Pittala Sandeep Kumar**



- GitHub: \[sandeepkumarpittala](https://github.com/sandeepkumarpittala)

- LinkedIn: \[Sandeep Kumar](https://www.linkedin.com/in/sandeep-kumar-pittala-06025b363/)
