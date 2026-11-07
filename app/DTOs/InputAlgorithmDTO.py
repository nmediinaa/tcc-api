from pydantic import BaseModel, Field

class RecommendRequestDTO(BaseModel):
    answers: list[float] = Field(..., min_length=23, max_length=23)