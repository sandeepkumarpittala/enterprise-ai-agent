from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    total_questions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    retrieval_hit_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    context_precision: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    context_recall: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    answer_correctness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    citation_correctness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    failure_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    average_latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )