"""Schémas Pydantic partagés par les agents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["support_technique", "facturation", "reclamation", "demande_info", "commercial", "rh_interne"]
Priority = Literal["haute", "moyenne", "basse"]
Sender = Literal["client", "interne", "fournisseur"]

INTENT_LABELS: list[str] = list(Intent.__args__)
PRIORITY_LABELS: list[str] = list(Priority.__args__)
SENDER_LABELS: list[str] = list(Sender.__args__)


class Email(BaseModel):
    id: str
    sender_email: str
    subject: str
    body: str


class ClassifierOutput(BaseModel):
    intent: Intent
    sender: Sender
    confidence: float = Field(ge=0, le=1, description="Confiance entre 0 et 1")
    rationale: str = Field(description="Justification courte (1 phrase)")


class PriorityOutput(BaseModel):
    priority: Priority
    urgency_cues: list[str] = Field(default_factory=list, description="Indices d'urgence relevés dans l'email")
    rationale: str


class ReviewOutput(BaseModel):
    approved: bool = Field(description="True si la classification est cohérente avec l'email")
    corrected_intent: Intent | None = None
    corrected_priority: Priority | None = None
    comment: str


class TriageResult(BaseModel):
    id: str
    intent: Intent
    priority: Priority
    sender: Sender
    confidence: float
    needs_human: bool
    trace: list[str]
