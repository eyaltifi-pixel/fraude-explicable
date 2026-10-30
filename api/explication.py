"""Outils d'explication partagés par le notebook (Module 5) et l'API (Module 6).

Aucune dépendance à SHAP ici : on reçoit des contributions déjà calculées
(contribution positive = la variable augmente le score d'anomalie = rend la
transaction plus suspecte) et on les regroupe / traduit en « codes motifs ».
"""
import numpy as np
import pandas as pd

# Les 9 variables du modèle sont regroupées en 8 notions lisibles par un analyste.
# (hour_sin et hour_cos forment ensemble UNE information : l'heure.
#  Les valeurs SHAP étant additives, on peut les additionner sans perte.)
GROUPES = {
    "Montant": ["log_amt"],
    "Population de la ville": ["city_pop"],
    "Distance client-marchand": ["distance_km"],
    "Heure": ["hour_sin", "hour_cos"],
    "Transactions sur 1 h": ["tx_1h"],
    "Transactions sur 24 h": ["tx_24h"],
    "Montant / moyenne 30 j": ["amt_ratio_30d"],
    "Délai depuis la précédente": ["secs_since_prev"],
}

CODES = {
    "Montant": "R01_MONTANT",
    "Population de la ville": "R02_VILLE",
    "Distance client-marchand": "R03_DISTANCE",
    "Heure": "R04_HEURE",
    "Transactions sur 1 h": "R05_VELOCITE_1H",
    "Transactions sur 24 h": "R06_VELOCITE_24H",
    "Montant / moyenne 30 j": "R07_ECART_PROFIL",
    "Délai depuis la précédente": "R08_DELAI",
}


def regrouper(values, X_num, num):
    """Regroupe les contributions par notion lisible.

    values : (n, 9) contributions dans l'ordre de `num`
    X_num  : DataFrame (n, 9) des variables du modèle (avant standardisation)
    Retourne (V, D, noms) :
      V = contributions regroupées (n, 8)
      D = valeurs lisibles associées (DataFrame n x 8 : montant en $, heure en h, ...)
      noms = liste des 8 notions
    """
    idx = {c: i for i, c in enumerate(num)}
    V = np.column_stack([values[:, [idx[c] for c in cols]].sum(axis=1)
                         for cols in GROUPES.values()])
    # heure d'origine à partir de (sin, cos) : angle = 2*pi*h/24 -> h = degrés / 15
    h = (np.degrees(np.arctan2(X_num["hour_sin"].values, X_num["hour_cos"].values)) / 15) % 24
    D = pd.DataFrame({
        "Montant": np.expm1(X_num["log_amt"].values),
        "Population de la ville": X_num["city_pop"].values,
        "Distance client-marchand": X_num["distance_km"].values,
        "Heure": np.round(h) % 24,
        "Transactions sur 1 h": X_num["tx_1h"].values,
        "Transactions sur 24 h": X_num["tx_24h"].values,
        "Montant / moyenne 30 j": X_num["amt_ratio_30d"].values,
        "Délai depuis la précédente": X_num["secs_since_prev"].values,
    })
    return V, D, list(GROUPES)


def _libelle(nom, v):
    """Phrase lisible pour un analyste."""
    manquant = v is None or (isinstance(v, float) and np.isnan(v))
    if nom == "Montant":
        return f"Montant inhabituel : {v:,.0f} $"
    if nom == "Population de la ville":
        return f"Ville de taille atypique : {v:,.0f} habitants"
    if nom == "Distance client-marchand":
        return f"Marchand à {v:,.0f} km du domicile du client"
    if nom == "Heure":
        return f"Transaction à {int(v)} h, heure peu habituelle"
    if nom == "Transactions sur 1 h":
        return f"{int(v)} transaction(s) déjà faite(s) avec cette carte dans l'heure précédente"
    if nom == "Transactions sur 24 h":
        return f"{int(v)} transaction(s) déjà faite(s) avec cette carte sur 24 h"
    if nom == "Montant / moyenne 30 j":
        if manquant:
            return "Aucun historique sur 30 j pour situer ce montant"
        return f"Montant = {v:.1f} × la dépense moyenne de la carte sur 30 j"
    if nom == "Délai depuis la précédente":
        if manquant:
            return "Première transaction connue de la carte"
        if v < 120:
            return f"Seulement {v:.0f} s après la transaction précédente"
        if v < 3600:
            return f"{v / 60:.0f} min après la transaction précédente"
        return f"{v / 3600:.1f} h après la transaction précédente"
    return nom


def motifs(contribs, valeurs, noms, k_max=5):
    """Codes motifs : les (au plus) k_max notions qui AUGMENTENT le plus le score.

    On ne garde que les contributions positives : un motif doit expliquer
    pourquoi la transaction est suspecte, pas pourquoi elle serait normale.
    """
    contribs = np.asarray(contribs, dtype=float)
    ordre = [i for i in np.argsort(-contribs) if contribs[i] > 0][:k_max]
    return [{"code": CODES[noms[i]],
             "variable": noms[i],
             "libelle": _libelle(noms[i], valeurs[i]),
             "contribution": round(float(contribs[i]), 4)} for i in ordre]
