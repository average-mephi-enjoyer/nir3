# core/schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class SemanticMarkerItem(BaseModel):
    """
    Element of the structured semantic marker with evidential binding.
    Pydantic generates __init__ and validation automatically.
    """
    marker_name: str = Field(
        ..., 
        description="Marker type: organizational_alienation, concealment_intent, rationalization, access_exfiltration_focus, conflict_escalation"
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Degree of marker expression from 0.0 to 1.0"
    )
    evidence_span: str = Field(
        ..., 
        description="Exact quote from the text serving as the basis for the assessment"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Degree of confidence in the marker identification"
    )
    alternative_explanation: Optional[str] = Field(
        None, 
        description="Alternative explanation (deadline, stress, joke, technical issue)" 
    )

class LLMSemanticAnalysisResponse(BaseModel):
    """
    Validated response from the local LLM for a single analytical window or message.
    """
    markers: List[SemanticMarkerItem] = Field(default_factory=list)
    has_insufficient_context: bool = Field(
        False, 
        description="Flag indicating insufficient context for a reliable assessment"
    )
    overall_sentiment_drift: float = Field(
        0.0, 
        ge=-1.0, 
        le=1.0, 
        description="Sentiment drift assessment (-1: negative, 1: positive)"
    )
    schema_version: str = Field("1.0", description="Version of the ontology schema")
