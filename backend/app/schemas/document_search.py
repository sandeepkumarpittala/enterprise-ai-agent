
from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated


class DocumentSearchRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    limit: int = Field(default=5, ge=1, le=20)
