"""Génère un jeu de données synthétique d'emails professionnels annotés.

Chaque email reçoit trois étiquettes :
- intent   : intention principale de l'expéditeur
- priority : haute / moyenne / basse
- sender   : type d'expéditeur (client, interne, fournisseur)

Le générateur est volontairement bruité (formulations variées, emails mixtes,
indices d'urgence implicites) pour que la tâche ne soit pas triviale.
Usage : python data/generate_dataset.py --n 400 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

INTENTS = {
    "support_technique": {
        "subjects": ["Problème de connexion", "Bug sur l'application", "Erreur lors de l'export",
                     "Impossible d'accéder au tableau de bord", "Lenteurs depuis la mise à jour"],
        "bodies": [
            "Depuis ce matin, je n'arrive plus à me connecter à mon compte, le message 'erreur 500' s'affiche.",
            "L'export CSV plante systématiquement dès que je sélectionne plus d'un mois de données.",
            "Après la dernière mise à jour, l'application est devenue très lente et certaines pages ne chargent plus.",
            "Le bouton de validation ne réagit plus sur Firefox, pouvez-vous regarder ?",
            "Nos utilisateurs reçoivent un message d'erreur en ouvrant le module de reporting.",
        ],
    },
    "facturation": {
        "subjects": ["Question sur ma facture", "Double prélèvement", "Facture de septembre",
                     "Demande d'avoir", "Erreur de montant"],
        "bodies": [
            "Nous avons été prélevés deux fois pour la facture du mois dernier, merci de régulariser.",
            "Le montant de la facture ne correspond pas au devis signé, pouvez-vous vérifier ?",
            "Pourriez-vous m'envoyer la facture de septembre au format PDF pour notre comptabilité ?",
            "Je souhaite obtenir un avoir suite à la résiliation de l'option premium.",
            "Notre service comptable ne trouve pas le numéro de TVA sur la dernière facture.",
        ],
    },
    "reclamation": {
        "subjects": ["Mécontentement", "Réclamation commande", "Service inacceptable",
                     "Livraison non conforme", "Plainte"],
        "bodies": [
            "C'est la troisième fois que la livraison arrive en retard, la situation n'est plus acceptable.",
            "Le produit reçu ne correspond pas du tout à la commande, je demande un remboursement.",
            "Je suis très déçu par la réponse de votre support, personne ne m'a rappelé.",
            "Le matériel livré est endommagé, je souhaite déposer une réclamation officielle.",
            "Nous envisageons de résilier notre contrat si la qualité de service ne s'améliore pas.",
        ],
    },
    "demande_info": {
        "subjects": ["Renseignements", "Question sur vos offres", "Informations produit",
                     "Disponibilité", "Documentation"],
        "bodies": [
            "Pourriez-vous m'indiquer les fonctionnalités incluses dans l'offre standard ?",
            "Je souhaiterais savoir si votre solution est compatible avec notre ERP.",
            "Avez-vous une documentation technique sur l'API que je pourrais consulter ?",
            "Quels sont vos délais habituels de mise en service pour une PME ?",
            "Est-il possible d'héberger les données en France ?",
        ],
    },
    "commercial": {
        "subjects": ["Demande de devis", "Proposition de partenariat", "Renouvellement de contrat",
                     "Extension de licence", "Rendez-vous commercial"],
        "bodies": [
            "Nous aimerions recevoir un devis pour 50 licences supplémentaires.",
            "Nous serions intéressés par un partenariat de distribution sur la région Ouest.",
            "Notre contrat arrive à échéance, pouvons-nous discuter des conditions de renouvellement ?",
            "Pouvez-vous nous proposer un créneau pour une démonstration à notre équipe ?",
            "Nous souhaitons étendre l'abonnement à nos filiales, quel serait le tarif ?",
        ],
    },
    "rh_interne": {
        "subjects": ["Demande de congés", "Note de frais", "Planning de l'équipe",
                     "Formation", "Entretien annuel"],
        "bodies": [
            "Je souhaiterais poser mes congés du 12 au 23 août, merci de valider.",
            "Vous trouverez ci-joint ma note de frais pour le déplacement à Lyon.",
            "Peux-tu mettre à jour le planning d'astreinte de la semaine prochaine ?",
            "Je voudrais m'inscrire à la formation Python avancé du mois prochain.",
            "Pouvons-nous fixer la date de mon entretien annuel ?",
        ],
    },
}

URGENT_CUES = [
    "C'est urgent, merci de traiter ce point aujourd'hui.",
    "Toute notre production est bloquée.",
    "Sans réponse sous 24h, nous serons contraints d'escalader.",
    "Le client final attend une réponse ce soir.",
]
SOFT_URGENT_CUES = [  # urgence implicite, plus difficile
    "Nous avons une démonstration importante demain matin.",
    "Notre clôture comptable a lieu cette semaine.",
]
LOW_CUES = [
    "Rien de pressé, quand vous aurez un moment.",
    "Pas d'urgence, c'est pour information.",
    "Vous pouvez me répondre la semaine prochaine.",
]
SIGNATURES = {
    "client": ["Cordialement,\n{name}\nResponsable achats, {company}", "Bien à vous,\n{name} ({company})"],
    "fournisseur": ["Cordialement,\n{name}\nService commercial, {company}", "{name}\nCompte clé, {company}"],
    "interne": ["Merci,\n{name}", "À plus,\n{name}\nÉquipe Data"],
}
NAMES = ["Claire Martin", "Hugo Bernard", "Amina Diallo", "Lucas Petit", "Sofia Rossi", "Yann Le Gall",
         "Nadia Benali", "Thomas Moreau", "Inès Garcia", "Paul Nguyen"]
COMPANIES = ["Atlantis SA", "Brest Logistique", "Nova Retail", "Kermor Industrie", "Helio Conseil",
             "DataSud", "Armor Tech", "Ouest Distribution"]

# Expéditeur plausible pour chaque intention (pour la cohérence des données)
SENDER_BY_INTENT = {
    "support_technique": ["client", "client", "interne"],
    "facturation": ["client", "fournisseur"],
    "reclamation": ["client"],
    "demande_info": ["client", "client", "fournisseur"],
    "commercial": ["client", "fournisseur"],
    "rh_interne": ["interne"],
}


def make_email(rng: random.Random, idx: int) -> dict:
    intent = rng.choice(list(INTENTS))
    sender = rng.choice(SENDER_BY_INTENT[intent])
    spec = INTENTS[intent]
    subject = rng.choice(spec["subjects"])
    body_idx = rng.randrange(len(spec["bodies"]))
    body = [spec["bodies"][body_idx]]

    # Bruit : 20 % des emails mélangent une phrase d'une autre intention
    if rng.random() < 0.20:
        other = rng.choice([i for i in INTENTS if i != intent])
        body.append("Par ailleurs, " + rng.choice(INTENTS[other]["bodies"]).lower())

    # Priorité dérivée de l'intention + indices d'urgence
    r = rng.random()
    if intent == "reclamation" or r < 0.25:
        if rng.random() < 0.3:
            body.append(rng.choice(SOFT_URGENT_CUES))
        else:
            body.append(rng.choice(URGENT_CUES))
        priority = "haute"
    elif r < 0.55 or intent == "rh_interne":
        body.append(rng.choice(LOW_CUES))
        priority = "basse"
    else:
        priority = "moyenne"

    name, company = rng.choice(NAMES), rng.choice(COMPANIES)
    signature = rng.choice(SIGNATURES[sender]).format(name=name, company=company)
    domain = "notre-entreprise.fr" if sender == "interne" else company.lower().replace(" ", "-") + ".com"
    sender_email = name.split()[0].lower() + "@" + domain
    greeting = rng.choice(["Bonjour,", "Bonjour Madame, Monsieur,", "Hello,", "Bonjour à tous,"])
    text = f"{greeting}\n\n" + " ".join(body) + f"\n\n{signature}"
    return {
        "id": f"email_{idx:04d}",
        "sender_email": sender_email,
        "subject": subject,
        "body": text,
        "intent": intent,
        "priority": priority,
        "sender": sender,
        # Identifiant du gabarit de phrase principal : sert au split groupé (anti-fuite)
        "template_id": f"{intent}_{body_idx}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "emails.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    df = pd.DataFrame([make_email(rng, i) for i in range(args.n)])
    df.to_csv(args.out, index=False)
    print(f"{len(df)} emails écrits dans {args.out}")
    print(df[["intent", "priority", "sender"]].describe())


if __name__ == "__main__":
    main()
