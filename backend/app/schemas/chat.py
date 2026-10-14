from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class ChatRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=2000,
        ),
    ]

    limit: int = Field(default=5, ge=1, le=10)