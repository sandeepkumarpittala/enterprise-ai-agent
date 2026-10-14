import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.schemas.chat import ChatRequest
from app.services.rag_service import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_chunks: int
    conversation_id: int


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    conversation_id: int | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    try:
        # Use the requested conversation only if it belongs to this user.
        conversation = None

        if conversation_id is not None:
            conversation = db.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == current_user.id,
                )
            )

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

        # Create a conversation automatically for a new chat.
        if conversation is None:
            title = request.question.strip()[:80]

            conversation = Conversation(
                user_id=current_user.id,
                title=title,
            )
            db.add(conversation)
            db.flush()

        # Generate a grounded answer from the user's documents.
        result = answer_question(
            question=request.question,
            user_id=current_user.id,
            limit=request.limit,
        )

        # Save both messages to the same conversation.
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=result["question"],
        )

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result["answer"],
        )

        db.add_all([user_message, assistant_message])
        db.flush()

        # Update the conversation's activity timestamp.
        conversation.updated_at = user_message.created_at

        db.commit()

        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            retrieved_chunks=result["retrieved_chunks"],
            conversation_id=conversation.id,
        )

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        db.rollback()
        logger.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=503,
            detail="The answer service is temporarily unavailable.",
        ) from exc

    except Exception as exc:
        db.rollback()
        logger.exception("Unexpected chat endpoint failure")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while answering the question.",
        ) from exc