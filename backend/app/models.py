from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID

class Document(BaseModel):
    id: UUID
    title: str
    content: str
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    confidence: float
    source_documents: List[Dict[str, Any]]  # Enhanced source document information