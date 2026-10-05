"""Évaluation : baseline TF-IDF vs pipeline multi-agents LLM.

Usage :
    python -m email_agents.evaluate                 # baseline seule (pas de clé API nécessaire)
    python -m email_agents.evaluate --agents --limit 60   # + agents LLM sur 60 emails de test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score)
from sklearn.model_selection import train_test_split

from .baseline import TARGETS, TfidfBaseline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "emails.csv"
REPORTS = ROOT / "reports"


def scores(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    return {t: {"accuracy": round(accuracy_score(y_true[t], y_pred[t]), 3),
                "macro_f1": round(f1_score(y_true[t], y_pred[t], average="macro"), 3)} for t in TARGETS}


def save_confusion(y_true, y_pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def split(df: pd.DataFrame, mode: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """random : split classique. grouped : les gabarits de phrases du test sont absents du train.

    Avec des emails générés à partir de gabarits, un split aléatoire place des quasi-doublons
    dans le train et le test (fuite de données) et surestime fortement la performance.
    """
    if mode == "random":
        return train_test_split(df, test_size=0.25, random_state=seed, stratify=df["intent"])
    # Pour chaque intention, un gabarit entier est réservé au test : toutes les classes sont
    # présentes des deux côtés, mais avec des formulations jamais vues à l'entraînement.
    rng = np.random.default_rng(seed)
    held_out = [rng.choice(sorted(g["template_id"].unique())) for _, g in df.groupby("intent")]
    is_test = df["template_id"].isin(held_out)
    return df[~is_test], df[is_test]


def run_agents(test: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, dict]:
    from dotenv import load_dotenv

    from .agents import EmailTriagePipeline
    from .llm import LLMClient
    from .schemas import Email

    load_dotenv()
    llm = LLMClient()
    pipe = EmailTriagePipeline(llm)
    rows = []
    for _, r in test.head(limit).iterrows():
        res = pipe.run(Email(id=r["id"], sender_email=r["sender_email"], subject=r["subject"], body=r["body"]))
        rows.append(res.model_dump())
        print(f"{res.id}: {res.intent:18s} {res.priority:8s} human={res.needs_human}")
    pred = pd.DataFrame(rows).set_index(test.head(limit).index)
    extra = {"llm_calls": llm.calls, "llm_calls_per_email": round(llm.calls / len(pred), 2),
             "needs_human_rate": round(pred["needs_human"].mean(), 3), "model": llm.model}
    return pred, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", action="store_true", help="Évaluer aussi le pipeline LLM (clé API requise)")
    ap.add_argument("--limit", type=int, default=60, help="Nombre d'emails de test pour les agents")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["grouped", "random"], default="grouped")
    args = ap.parse_args()

    REPORTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    # 0) Démonstration de la fuite : baseline sur split aléatoire
    tr_r, te_r = split(df, "random", args.seed)
    leaky = scores(te_r, TfidfBaseline().fit(tr_r).predict(te_r))

    train, test = split(df, args.split, args.seed)
    results: dict = {"split": args.split, "n_train": len(train), "n_test": len(test),
                     "baseline_random_split_leaky": leaky}

    # 1) Baseline
    base = TfidfBaseline().fit(train)
    base_pred = base.predict(test)
    results["baseline_tfidf_logreg"] = scores(test, base_pred)
    save_confusion(test["intent"], base_pred["intent"], "Baseline TF-IDF - intention",
                   REPORTS / "confusion_baseline_intent.png")
    print("=== Baseline TF-IDF + LogReg ===")
    for t in TARGETS:
        print(f"\n[{t}]\n" + classification_report(test[t], base_pred[t], zero_division=0))

    # 2) Agents LLM
    if args.agents:
        sub = test.head(args.limit)
        agent_pred, extra = run_agents(test, args.limit)
        results["agents_llm"] = {**scores(sub, agent_pred), **extra}
        results["baseline_on_same_subset"] = scores(sub, base_pred.loc[sub.index])
        save_confusion(sub["intent"], agent_pred["intent"], "Agents LLM - intention",
                       REPORTS / "confusion_agents_intent.png")
        errors = sub.join(agent_pred[["intent", "priority", "needs_human", "trace"]], rsuffix="_pred")
        errors = errors[(errors["intent"] != errors["intent_pred"]) | (errors["priority"] != errors["priority_pred"])]
        errors.to_csv(REPORTS / "agent_errors.csv", index=False)
        print(f"\n{len(errors)} erreurs des agents enregistrées dans reports/agent_errors.csv")

    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\nRésumé :\n" + json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
