from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator, ValidationInfo

RiskMarkerType = Literal[
    "organizational_alienation",
    "concealment_intent",
    "rationalization",
    "access_exfiltration_focus",
    "conflict_escalation"
]


class SemanticMarkerItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    marker_name: RiskMarkerType = Field(
        ..., 
        description="Author-defined semantic category, not an original ILI item"
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Grade of degeneracy (0.0 - no risk, 1.0 - risk)"
    )
    evidence_span: str = Field(
        ..., 
        min_length=3,
        description="Citation of evidence"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence of the model (1.0 - max, 0.0 - min)"
    )
    alternative_explanation: Optional[str] = Field(
        default=None, 
        description="Alternative explanation"
    )


class LLMSemanticAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    markers: List[SemanticMarkerItem] = Field(
        default_factory=list,
        description="Semantic markers list"
    )
    has_insufficient_context: bool = Field(
        default=False, 
        description="Content insufficiency flag"
    )
    overall_sentiment_drift: float = Field(
        default=0.0, 
        ge=-1.0, 
        le=1.0, 
        description="Segment grade: -1.0 (destructive) до +1.0 (constructive)"
    )
    schema_version: str = Field(
        default="1.1", 
        description="Schema version"
    )

    @model_validator(mode="after")
    def validate_evidence(self, info: ValidationInfo):
        names = [item.marker_name for item in self.markers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate semantic markers")
        if info.context and "text" in info.context:
            text = info.context["text"]
            for item in self.markers:
                if item.evidence_span not in text:
                    raise ValueError("Evidence is not an exact span of the analyzed text")
        return self
