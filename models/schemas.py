from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class TriageOutput(BaseModel):
    """Structured security triage schema."""
    status: Literal["alert", "clear"] = Field(
        description="Whether this log sequence represents an actual threat or benign noise."
    )
    category: Literal["security", "system_health", "noise"] = Field(
        description="Categorization of the event."
    )
    mitre_id: Optional[str] = Field(
        default=None, 
        description="MITRE ATT&CK ID if relevant (e.g. T1059.004), null otherwise."
    )
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(
        description="Concise rationale explaining the verdict and whether RAG context was accepted or rejected."
    )
    recommended_action: Optional[str] = Field(
        default=None, 
        description="Active response or remediation suggestion."
    )


class MitreTechnique(BaseModel):
    """Schema representing an extracted MITRE ATT&CK technique."""
    mitre_id: str = Field(description="The MITRE ATT&CK technique identifier, e.g. T1059.001")
    name: str = Field(description="The name of the technique")
    description: str = Field(default="", description="Detailed description of the technique")


# Alias for backward compatibility if referenced in legacy modules
MitreSchema = MitreTechnique


class AgentState(TypedDict):
    """Shared state across all LangGraph nodes."""
    raw_logs: str
    search_query: str
    retrieved_chunks: List[str]
    triage_result: Optional[TriageOutput]
    retry_count: int
    error_msg: Optional[str]