# api/models/responses.py
from typing import Dict, Any, Union
from pydantic import BaseModel, field_validator
from knowledge_graph.models import KnowledgeGraphResponse


class ProcessPromptResponse(BaseModel):
    """
    FastAPI response model that accepts both KnowledgeGraphResponse objects and dicts
    """

    status: str
    knowledge_graph: Union[KnowledgeGraphResponse, Dict[str, Any]]
    message: str

    @field_validator("knowledge_graph", mode="before")
    @classmethod
    def validate_knowledge_graph(cls, v):
        """Convert KnowledgeGraphResponse to dict for serialization"""
        if isinstance(v, KnowledgeGraphResponse):
            return v.model_dump()
        return v
