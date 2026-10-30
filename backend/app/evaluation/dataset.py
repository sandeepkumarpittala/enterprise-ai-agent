
"""Version-controlled dataset for evaluating RAG behavior."""

EVALUATION_DATASET = [
    {
        "id": "rag-001",
        "question": "Which vector database does the application use?",
        "expected_answer": "Qdrant",
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [14, 15],
    },
    {
        "id": "rag-002",
        "question": "Which framework powers the backend?",
        "expected_answer": "FastAPI",
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [14, 15],
    },
    {
        "id": "rag-003",
        "question": "Which embedding model is used?",
        "expected_answer": "sentence-transformers/all-MiniLM-L6-v2",
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [14, 15, 16],
    },
    {
        "id": "rag-004",
        "question": "What are the steps in document indexing?",
        "expected_answer": (
            "Extract document text, split it into chunks, generate "
            "embeddings, and store them with text and metadata."
        ),
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [13, 14, 15, 16],
    },
    {
        "id": "rag-005",
        "question": "What is the name of the fictional company director?",
        "expected_answer": None,
        "expected_source": None,
        "answerable": False,
        "relevant_chunk_ids": [],
    },
    {
        "id": "rag-006",
        "question": (
            "Which frontend technology and build tool does the application use?"
        ),
        "expected_answer": "React with Vite",
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [14, 15],
    },
    {
        "id": "rag-007",
        "question": "Which relational database stores application data?",
        "expected_answer": "PostgreSQL",
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [14, 15],
    },
    {
        "id": "rag-008",
        "question": "Which tool runs the language model locally?",
        "expected_answer": "Ollama",
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [14, 15],
    },
    {
        "id": "rag-009",
        "question": "How does the application answer a user's question?",
        "expected_answer": (
            "It embeds the question, searches Qdrant for similar chunks, "
            "passes the question and retrieved context to the language model, "
            "and generates an answer."
        ),
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [14, 15],
    },
    {
        "id": "rag-010",
        "question": (
            "What should I check if the answer is in the PDF "
            "but the agent cannot find it?"
        ),
        "expected_answer": (
            "Check whether text extraction succeeded, chunks were created, "
            "embeddings were stored, and retrieval returns the expected chunks. "
            "Also confirm that the document belongs to the authenticated user."
        ),
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [15],
    },
    {
        "id": "rag-011",
        "question": "How does RAG help reduce unsupported answers?",
        "expected_answer": (
            "RAG retrieves relevant information from a knowledge source "
            "and provides it to a language model to generate a grounded answer."
        ),
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [13],
    },
    {
        "id": "rag-012",
        "question": (
            "Which vector database and relational database does the "
            "application use?"
        ),
        "expected_answer": "Qdrant and PostgreSQL",
        "expected_source": "enterprise_ai_rag_test_document.pdf",
        "answerable": True,
        "relevant_chunk_ids": [14, 15],
    },
    {
        "id": "rag-013",
        "question": "What is the name of the company's real-world CEO?",
        "expected_answer": None,
        "expected_source": None,
        "answerable": False,
        "relevant_chunk_ids": [],
    },
    {
        "id": "rag-014",
        "question": (
            "What is the exact number of employees at the fictional company?"
        ),
        "expected_answer": None,
        "expected_source": None,
        "answerable": False,
        "relevant_chunk_ids": [],
    },
    {
        "id": "rag-015",
        "question": (
            "What is the exact monthly operating cost of the application?"
        ),
        "expected_answer": None,
        "expected_source": None,
        "answerable": False,
        "relevant_chunk_ids": [],
    },
]
