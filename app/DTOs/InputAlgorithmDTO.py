from pydantic import BaseModel, Field

class RecommendRequestDTO(BaseModel):
    answers: list[float] = Field(..., min_length=15, max_length=15)