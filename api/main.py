"""API FastAPI : détection de fraude explicable (Isolation Forest, non supervisé).

Lancement (depuis le dossier api/) :
    uvicorn main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from explication import regrouper, motifs  # noqa: E402

# --------------------------------------------------------------------------
# Chargement des artefacts sauvegardés par le notebook (une seule fois)
# --------------------------------------------------------------------------
ART = joblib.load(Path(__file__).parent / "artefacts.joblib")
ISO, IMP, SC = ART["iso"], ART["imputer"], ART["scaler"]
NUM, SKEW = ART["num"], ART["skew"]
SEUIL_COUT, SEUIL_BUDGET = ART["seuil_cout"], ART["seuil_budget"]
BG = ART["background"]


def _prep(D: pd.DataFrame) -> pd.DataFrame:
    """Même transformation que le notebook : log des variables asymétriques."""
    Z = D[NUM].astype("float32").copy()
    Z[SKEW] = np.log1p(Z[SKEW])
    return Z


def _score(A: np.ndarray) -> np.ndarray:
    """Score d'anomalie (élevé = suspect) à partir des 9 variables brutes."""
    D = pd.DataFrame(A, columns=NUM)
    return -ISO.score_samples(SC.transform(IMP.transform(_prep(D))))


# Explicateur SHAP indépendant du modèle : il explique directement le score
# d'anomalie, donc contribution positive = plus suspect.
EXPLAINER = shap.Explainer(_score, shap.maskers.Independent(BG, max_samples=50),
                           algorithm="permutation", feature_names=NUM)

# --------------------------------------------------------------------------
# Schémas d'entrée / sortie
# --------------------------------------------------------------------------


class Transaction(BaseModel):
    amt: float = Field(..., ge=0, description="Montant de la transaction ($)")
    city_pop: int = Field(..., ge=0, description="Population de la ville du client")
    distance_km: float = Field(..., ge=0, description="Distance client-marchand (km)")
    heure: int = Field(..., ge=0, le=23, description="Heure de la transaction (0-23)")
    tx_1h: int = Field(0, ge=0, description="Transactions de la carte dans l'heure précédente")
    tx_24h: int = Field(0, ge=0, description="Transactions de la carte sur les 24 h précédentes")
    amt_ratio_30d: Optional[float] = Field(
        None, ge=0, description="Montant / dépense moyenne 30 j (null si pas d'historique)")
    secs_since_prev: Optional[float] = Field(
        None, ge=0, description="Secondes depuis la transaction précédente (null si première)")

    model_config = {"json_schema_extra": {"example": {
        "amt": 842.5, "city_pop": 1200, "distance_km": 74.2, "heure": 2,
        "tx_1h": 2, "tx_24h": 4, "amt_ratio_30d": 9.8, "secs_since_prev": 95}}}


class Prediction(BaseModel):
    score: float
    niveau: str            # "normal" | "alerte" | "critique"
    alerte: bool
    seuil_cout: float
    seuil_budget: float


class Explication(Prediction):
    valeur_de_base: float
    contributions: List[dict]
    codes_motifs: List[dict]


def _vers_matrice(transactions: List[Transaction]) -> np.ndarray:
    lignes = []
    for t in transactions:
        angle = 2 * np.pi * t.heure / 24
        lignes.append([
            np.log1p(t.amt), t.city_pop, t.distance_km, np.sin(angle), np.cos(angle),
            t.tx_1h, t.tx_24h,
            np.nan if t.amt_ratio_30d is None else t.amt_ratio_30d,
            np.nan if t.secs_since_prev is None else t.secs_since_prev])
    return np.array(lignes, dtype=float)   # ordre = NUM


def _niveau(s: float) -> str:
    if s >= SEUIL_BUDGET:
        return "critique"      # dans le budget d'alertes de l'analyste
    if s >= SEUIL_COUT:
        return "alerte"        # au-dessus du seuil optimal en coût
    return "normal"


def _prediction(s: float) -> Prediction:
    return Prediction(score=round(float(s), 6), niveau=_niveau(s),
                      alerte=bool(s >= SEUIL_COUT),
                      seuil_cout=SEUIL_COUT, seuil_budget=SEUIL_BUDGET)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
app = FastAPI(title="API de détection de fraude explicable",
              description="Isolation Forest (non supervisé) + explications SHAP.",
              version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"modele": ART["modele"], "type": "Isolation Forest (non supervisé)",
            "variables": NUM, "seuil_cout": SEUIL_COUT, "seuil_budget": SEUIL_BUDGET,
            "convention": "score élevé = transaction plus suspecte"}


@app.post("/predict", response_model=Prediction)
def predict(t: Transaction):
    return _prediction(_score(_vers_matrice([t]))[0])


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(transactions: List[Transaction]):
    if not 0 < len(transactions) <= 5000:
        raise HTTPException(status_code=422, detail="Fournir entre 1 et 5000 transactions.")
    return [_prediction(s) for s in _score(_vers_matrice(transactions))]


@app.post("/explain", response_model=Explication)
def explain(t: Transaction):
    X = _vers_matrice([t])
    sv = EXPLAINER(X, max_evals=10 * len(NUM) + 1)
    X_df = pd.DataFrame(X, columns=NUM)
    V, D, noms = regrouper(sv.values, X_df, NUM)
    valeurs = D.iloc[0].values
    contrib = [{"variable": n, "valeur": None if pd.isna(v) else float(v), "contribution": round(float(c), 4)}
               for n, v, c in zip(noms, valeurs, V[0])]
    contrib.sort(key=lambda d: -d["contribution"])
    pred = _prediction(_score(X)[0])
    return Explication(**pred.model_dump(), valeur_de_base=float(sv.base_values[0]),
                       contributions=contrib, codes_motifs=motifs(V[0], valeurs, noms))
