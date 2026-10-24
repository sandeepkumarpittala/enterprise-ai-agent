
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.auth import get_current_user, get_db
from app.main import app


USER_A = SimpleNamespace(id=1, email="user-a@example.com", is_active=True)
USER_B = SimpleNamespace(id=2, email="user-b@example.com", is_active=True)


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[get_current_user] = lambda: USER_A

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_conversation_messages_are_scoped_to_current_user(client):
    """A user must not retrieve another user's conversation."""
    with patch("app.api.conversations.select") as mock_select:
        mock_statement = MagicMock()
        mock_statement.where.return_value = mock_statement
        mock_select.return_value = mock_statement

        mock_db = MagicMock()
        mock_db.scalar.return_value = None

        app.dependency_overrides[get_db] = lambda: mock_db

        response = client.get("/conversations/21/messages")

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."
    mock_db.scalar.assert_called_once()


def test_document_listing_uses_current_user(client):
    """The document listing must filter by the authenticated user."""
    mock_db = MagicMock()
    mock_db.scalars.return_value.all.return_value = []
    app.dependency_overrides[get_db] = lambda: mock_db

    with patch("app.api.documents.select") as mock_select:
        mock_statement = MagicMock()
        mock_statement.where.return_value = mock_statement
        mock_statement.order_by.return_value = mock_statement
        mock_select.return_value = mock_statement

        response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"count": 0, "documents": []}
    mock_statement.where.assert_called_once()


def test_document_search_passes_current_user_id(client):
    """Search must be scoped to the authenticated user's documents."""
    with patch(
        "app.api.documents.search_documents",
        return_value=[],
    ) as mock_search:
        response = client.post(
            "/documents/search",
            json={
                "query": "private document",
                "limit": 5,
            },
        )

    assert response.status_code == 200
    assert response.json()["results"] == []
    mock_search.assert_called_once_with(
        query="private document",
        user_id=USER_A.id,
        limit=5,
    )


def test_delete_other_users_document_is_denied(client):
    """A user must not delete a document they do not own."""
    mock_db = MagicMock()
    mock_db.scalar.return_value = None
    app.dependency_overrides[get_db] = lambda: mock_db

    with patch("app.api.documents.select") as mock_select:
        mock_statement = MagicMock()
        mock_statement.where.return_value = mock_statement
        mock_select.return_value = mock_statement

        response = client.delete("/documents/14")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."
    mock_db.delete.assert_not_called()
    mock_db.commit.assert_not_called()


def test_chat_rejects_another_users_conversation(client):
    """Chat must not append messages to another user's conversation."""
    mock_db = MagicMock()
    mock_db.scalar.return_value = None
    app.dependency_overrides[get_db] = lambda: mock_db

    response = client.post(
        "/chat?conversation_id=21",
        json={
            "question": "What vector database does this project use?",
            "limit": 5,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()
