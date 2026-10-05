"""Classer un email depuis la ligne de commande.

    python -m email_agents.cli --subject "Double prélèvement" --body "Nous avons été prélevés deux fois..." \
        --sender client@acme.com
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .agents import EmailTriagePipeline
from .llm import LLMClient
from .schemas import Email


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--sender", default="inconnu@exemple.com")
    args = ap.parse_args()
    load_dotenv()
    res = EmailTriagePipeline(LLMClient()).run(
        Email(id="cli", sender_email=args.sender, subject=args.subject, body=args.body))
    print(res.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
