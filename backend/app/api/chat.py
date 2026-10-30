
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
        conversation = None

        # Verify that the conversation belongs to the authenticated user.
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

        # Create a conversation when no ID is provided.
        if conversation is None:
            conversation = Conversation(
                user_id=current_user.id,
                title=request.question.strip()[:80],
            )
            db.add(conversation)
            db.flush()

        # Load the latest 10 messages in chronological order.
        history: list[dict[str, str]] = []

        if conversation_id is not None:
            previous_messages = db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(
                    Message.created_at.desc(),
                    Message.id.desc(),
                )
                .limit(10)
            ).all()

            history = [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in reversed(previous_messages)
            ]

        # Generate an answer using document context and conversation history.
        result = answer_question(
            question=request.question,
            user_id=current_user.id,
            limit=request.limit,
            conversation_history=history,
        )

        # Save both the user's question and the assistant's answer.
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=result["question"],
        )

        assistant_message = Message(
    conversation_id=conversation.id,
    role="assistant",
    content=result["answer"],
    sources=result.get("sources", []),
)
        db.add_all([user_message, assistant_message])
        db.flush()

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
            detail=(
                "An unexpected error occurred while answering the question."
            ),
        ) from exc
