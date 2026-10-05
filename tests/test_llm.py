"""Teste la validation JSON et l'auto-correction du client LLM, sans réseau."""
from types import SimpleNamespace

import pytest

from email_agents import llm as llm_mod
from email_agents.schemas import PriorityOutput


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.messages_seen = []

    def create(self, **kwargs):
        self.messages_seen.append(kwargs["messages"])
        msg = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_client(monkeypatch, contents):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    client = llm_mod.LLMClient(provider="openai")
    fake = FakeCompletions(contents)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


def test_valid_json(monkeypatch):
    client, _ = make_client(monkeypatch, ['{"priority": "haute", "urgency_cues": ["urgent"], "rationale": "x"}'])
    out = client.chat_json("sys", "user", PriorityOutput)
    assert out.priority == "haute" and client.calls == 1


def test_invalid_json_is_corrected(monkeypatch):
    client, fake = make_client(monkeypatch, ['{"priority": "critique", "rationale": "x"}',
                                             '{"priority": "haute", "rationale": "x"}'])
    out = client.chat_json("sys", "user", PriorityOutput)
    assert out.priority == "haute" and client.calls == 2
    assert "JSON invalide" in fake.messages_seen[1][-1]["content"]


def test_gives_up_after_retries(monkeypatch):
    client, _ = make_client(monkeypatch, ["pas du json"] * 3)
    with pytest.raises(RuntimeError):
        client.chat_json("sys", "user", PriorityOutput)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        llm_mod.LLMClient(provider="mistral")
