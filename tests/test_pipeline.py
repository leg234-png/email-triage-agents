"""Tests du pipeline sans appel réseau : un faux LLM renvoie des réponses scriptées."""
from __future__ import annotations

from collections import deque

import pandas as pd
import pytest

from email_agents.agents import EmailTriagePipeline
from email_agents.baseline import TfidfBaseline
from email_agents.schemas import ClassifierOutput, Email, PriorityOutput, ReviewOutput


class FakeLLM:
    """Renvoie, pour chaque schéma demandé, la prochaine réponse de sa file."""

    def __init__(self, scripted: dict[type, list]):
        self.queues = {k: deque(v) for k, v in scripted.items()}
        self.calls = []

    def chat_json(self, system, user, schema):
        self.calls.append((schema.__name__, user))
        return self.queues[schema].popleft()


EMAIL = Email(id="t1", sender_email="claire@atlantis-sa.com", subject="Double prélèvement",
              body="Nous avons été prélevés deux fois. C'est urgent.")


def cls(intent="facturation", conf=0.9):
    return ClassifierOutput(intent=intent, sender="client", confidence=conf, rationale="test")


def prio(p="haute"):
    return PriorityOutput(priority=p, urgency_cues=["urgent"], rationale="test")


def test_happy_path_approved():
    llm = FakeLLM({ClassifierOutput: [cls()], PriorityOutput: [prio()],
                   ReviewOutput: [ReviewOutput(approved=True, comment="ok")]})
    res = EmailTriagePipeline(llm).run(EMAIL)
    assert (res.intent, res.priority, res.needs_human) == ("facturation", "haute", False)
    assert [c[0] for c in llm.calls] == ["ClassifierOutput", "PriorityOutput", "ReviewOutput"]


def test_rejection_triggers_reclassification_with_feedback():
    llm = FakeLLM({
        ClassifierOutput: [cls("demande_info"), cls("facturation")],
        PriorityOutput: [prio("moyenne"), prio("haute")],
        ReviewOutput: [ReviewOutput(approved=False, corrected_intent="facturation", comment="c'est un prélèvement"),
                       ReviewOutput(approved=True, comment="ok")],
    })
    res = EmailTriagePipeline(llm).run(EMAIL)
    assert res.intent == "facturation" and not res.needs_human
    second_classify_prompt = [u for n, u in llm.calls if n == "ClassifierOutput"][1]
    assert "c'est un prélèvement" in second_classify_prompt  # le retour du reviewer est transmis


def test_two_rejections_escalate_to_human_with_correction():
    reject = ReviewOutput(approved=False, corrected_intent="reclamation", corrected_priority="haute", comment="non")
    llm = FakeLLM({ClassifierOutput: [cls("demande_info")] * 2, PriorityOutput: [prio("basse")] * 2,
                   ReviewOutput: [reject, reject]})
    res = EmailTriagePipeline(llm).run(EMAIL)
    assert res.needs_human
    assert (res.intent, res.priority) == ("reclamation", "haute")


def test_low_confidence_flags_human():
    llm = FakeLLM({ClassifierOutput: [cls(conf=0.4)], PriorityOutput: [prio()],
                   ReviewOutput: [ReviewOutput(approved=True, comment="ok")]})
    assert EmailTriagePipeline(llm).run(EMAIL).needs_human


def test_baseline_learns_simple_dataset():
    rows = []
    for i in range(30):
        rows.append(dict(sender_email="a@x.com", subject="Facture", body=f"facture prélèvement montant {i}",
                         intent="facturation", priority="moyenne", sender="client"))
        rows.append(dict(sender_email="b@notre-entreprise.fr", subject="Congés", body=f"poser mes congés {i}",
                         intent="rh_interne", priority="basse", sender="interne"))
    df = pd.DataFrame(rows)
    pred = TfidfBaseline().fit(df).predict(df)
    assert (pred["intent"] == df["intent"]).mean() == pytest.approx(1.0)
