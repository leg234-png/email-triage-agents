"""Pipeline multi-agents de tri d'emails, orchestré avec LangGraph.

Graphe :
    classify ──> prioritize ──> review ──┬─(approuvé)──────────────> finalize
        ^                                ├─(rejeté, 1er essai)──> classify (avec le retour du reviewer)
        └────────────────────────────────┘
                                         └─(rejeté, 2e essai)──> finalize (needs_human = True)

- ClassifierAgent : intention + type d'expéditeur, avec score de confiance
- PriorityAgent   : priorité à partir des indices d'urgence (explicites et implicites)
- ReviewerAgent   : contrôle la cohérence globale, peut corriger ou renvoyer en classification
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .llm import LLM
from .schemas import (ClassifierOutput, Email, PriorityOutput, ReviewOutput, TriageResult)

CONFIDENCE_THRESHOLD = 0.6
MAX_RECLASSIFY = 1

CLASSIFIER_PROMPT = """Tu es un agent de tri d'emails dans une entreprise B2B.
Classe l'email selon :
- intent : support_technique (bug, accès, erreur), facturation (facture, prélèvement, avoir),
  reclamation (mécontentement, plainte, menace de résiliation), demande_info (question sur l'offre,
  la compatibilité, la documentation), commercial (devis, partenariat, renouvellement, démo),
  rh_interne (congés, notes de frais, planning, formation).
- sender : client, interne (domaine notre-entreprise.fr) ou fournisseur.
Si l'email contient plusieurs sujets, choisis l'intention PRINCIPALE (en général la première demande).
Donne une confiance réaliste : baisse-la si l'email est ambigu."""

PRIORITY_PROMPT = """Tu es un agent qui fixe la priorité de traitement d'un email déjà classé.
- haute : blocage, réclamation, échéance proche (aujourd'hui, demain, cette semaine), menace d'escalade.
- basse : l'expéditeur indique explicitement que ce n'est pas urgent, ou demande interne de routine.
- moyenne : tout le reste.
Relève les indices d'urgence (citations courtes) avant de décider."""

REVIEWER_PROMPT = """Tu es un agent de contrôle qualité. Vérifie que l'intention et la priorité proposées
sont cohérentes avec l'email. Approuve si c'est correct. Sinon, propose une correction et explique
brièvement l'erreur. Sois exigeant mais n'invente pas de problème."""


class TriageState(TypedDict, total=False):
    email: Email
    classification: ClassifierOutput
    priority: PriorityOutput
    review: ReviewOutput
    feedback: str
    attempts: int
    trace: list[str]
    result: TriageResult


def _format_email(email: Email) -> str:
    return f"De : {email.sender_email}\nObjet : {email.subject}\n\n{email.body}"


class EmailTriagePipeline:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.graph = self._build_graph()

    # --- Nœuds -------------------------------------------------------------
    def classify(self, state: TriageState) -> TriageState:
        user = _format_email(state["email"])
        if state.get("feedback"):
            user += f"\n\nRetour du contrôle qualité sur ta proposition précédente : {state['feedback']}"
        out = self.llm.chat_json(CLASSIFIER_PROMPT, user, ClassifierOutput)
        trace = state.get("trace", []) + [f"classify: {out.intent}/{out.sender} (conf={out.confidence:.2f})"]
        return {"classification": out, "attempts": state.get("attempts", 0) + 1, "trace": trace}

    def prioritize(self, state: TriageState) -> TriageState:
        c = state["classification"]
        user = f"{_format_email(state['email'])}\n\nIntention détectée : {c.intent}"
        out = self.llm.chat_json(PRIORITY_PROMPT, user, PriorityOutput)
        return {"priority": out, "trace": state["trace"] + [f"prioritize: {out.priority} {out.urgency_cues}"]}

    def review(self, state: TriageState) -> TriageState:
        c, p = state["classification"], state["priority"]
        user = (f"{_format_email(state['email'])}\n\nProposition : intent={c.intent}, sender={c.sender}, "
                f"priority={p.priority}\nJustifications : {c.rationale} | {p.rationale}")
        out = self.llm.chat_json(REVIEWER_PROMPT, user, ReviewOutput)
        trace = state["trace"] + [f"review: {'OK' if out.approved else 'REJET'} - {out.comment}"]
        return {"review": out, "feedback": "" if out.approved else out.comment, "trace": trace}

    def finalize(self, state: TriageState) -> TriageState:
        c, p, r = state["classification"], state["priority"], state["review"]
        intent = c.intent if r.approved else (r.corrected_intent or c.intent)
        priority = p.priority if r.approved else (r.corrected_priority or p.priority)
        needs_human = (not r.approved) or c.confidence < CONFIDENCE_THRESHOLD
        result = TriageResult(id=state["email"].id, intent=intent, priority=priority, sender=c.sender,
                              confidence=c.confidence, needs_human=needs_human, trace=state["trace"])
        return {"result": result}

    # --- Routage -----------------------------------------------------------
    @staticmethod
    def route_after_review(state: TriageState) -> str:
        if state["review"].approved:
            return "finalize"
        if state.get("attempts", 0) <= MAX_RECLASSIFY:
            return "classify"
        return "finalize"

    def _build_graph(self):
        g = StateGraph(TriageState)
        g.add_node("classify", self.classify)
        g.add_node("prioritize", self.prioritize)
        g.add_node("review", self.review)
        g.add_node("finalize", self.finalize)
        g.set_entry_point("classify")
        g.add_edge("classify", "prioritize")
        g.add_edge("prioritize", "review")
        g.add_conditional_edges("review", self.route_after_review,
                                {"classify": "classify", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, email: Email) -> TriageResult:
        final = self.graph.invoke({"email": email, "trace": [], "attempts": 0})
        return final["result"]
