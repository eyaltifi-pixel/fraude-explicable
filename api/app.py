"""Application web de détection de fraude explicable (Streamlit).

Lancement (dossier api/, avec le serveur FastAPI déjà démarré dans un autre terminal) :
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Détection de fraude explicable", layout="wide")
st.title("Détection de fraude explicable")
st.caption("Isolation Forest (non supervisé) · seuil et budget réglables · explications SHAP")


@st.cache_data
def charger():
    return pd.read_csv("donnees_app.csv.gz")


df = charger()

# ------------------------------------------------------------------ barre latérale
st.sidebar.header("Paramètres")
api_url = st.sidebar.text_input("URL de l'API", "http://127.0.0.1:8000")
jeu = st.sidebar.radio("Jeu de données", ["validation", "test"])
c_alerte = st.sidebar.number_input("Coût d'analyse d'une alerte ($)", 0.5, 200.0, 10.0, step=0.5)

d = df[df["jeu"] == jeu].sort_values("score", ascending=False).reset_index(drop=True)
s, y, a = d["score"].values, d["fraude"].values, d["amt"].values
N, F = len(d), int(y.sum())

try:
    info = requests.get(f"{api_url}/model-info", timeout=5).json()
    seuil_defaut = float(info["seuil_cout"])
except Exception:
    info, seuil_defaut = None, float(np.quantile(s, 0.95))

seuil = st.sidebar.slider("Seuil de score", float(s.min()), float(s.max()),
                          float(np.clip(seuil_defaut, s.min(), s.max())), format="%.4f")
budget_pct = st.sidebar.slider("Budget d'alertes (% des transactions)", 0.1, 10.0, 1.0, step=0.1)
k_budget = int(budget_pct / 100 * N)

# ------------------------------------------------------------------ calculs en direct
# Une transaction est alertée si son score dépasse le seuil ET si elle rentre dans le budget.
k = min(int((s >= seuil).sum()), k_budget)
tp = int(y[:k].sum()); fp = k - tp; fn = F - tp; tn = N - k - fn
prec = tp / k if k else 0.0
rap = tp / F if F else 0.0
f2 = 5 * prec * rap / (4 * prec + rap) if (4 * prec + rap) > 0 else 0.0
den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
mcc = (tp * tn - fp * fn) / den if den else 0.0
montant_fraudes = float((a * y).sum())
cout = montant_fraudes - float((a * y)[:k].sum()) + c_alerte * k
economie = 1 - cout / montant_fraudes if montant_fraudes else 0.0

# ------------------------------------------------------------------ onglets
onglet1, onglet2 = st.tabs(["Seuil, budget et coût", "Expliquer une alerte"])

with onglet1:
    c = st.columns(6)
    c[0].metric("Alertes", f"{k:,}", f"{k / N:.2%} des transactions")
    c[1].metric("Précision", f"{prec:.1%}")
    c[2].metric("Rappel", f"{rap:.1%}")
    c[3].metric("F2", f"{f2:.3f}")
    c[4].metric("MCC", f"{mcc:.3f}")
    c[5].metric("Coût total ($)", f"{cout:,.0f}", f"économie {economie:.1%}")
    if int((s >= seuil).sum()) > k_budget:
        st.warning(f"Le seuil déclencherait {int((s >= seuil).sum()):,} alertes : "
                   f"le budget ({k_budget:,}) en limite le nombre. Il faudrait plus d'analystes pour les traiter toutes.")

    kmax = min(N, max(5 * k_budget, 2 * k, 1000))
    ks = np.arange(0, kmax + 1)
    cout_k = montant_fraudes - np.concatenate([[0], np.cumsum(a * y)])[: kmax + 1] + c_alerte * ks
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(ks, cout_k, label="coût total")
    ax.axhline(montant_fraudes, color="gray", ls=":", label="sans détecteur")
    ax.axvline(k, color="red", ls="--", label=f"réglage actuel (k={k:,})")
    ax.axvline(int(np.argmin(cout_k)), color="green", ls="--", label=f"optimum (k*={int(np.argmin(cout_k)):,})")
    ax.set_xlabel("Nombre d'alertes analysées"); ax.set_ylabel("Coût ($)"); ax.legend()
    st.pyplot(fig)
    st.caption(f"{N:,} transactions · {F:,} fraudes ({F / N:.2%}) · montant fraudé total : {montant_fraudes:,.0f} $")

with onglet2:
    if k == 0:
        st.info("Aucune alerte avec ce réglage : baisse le seuil.")
    else:
        nb = min(k, 200)
        st.write(f"Alertes les plus suspectes (rang 1 à {nb}) :")
        st.dataframe(d.head(nb)[["score", "amt", "heure", "distance_km", "tx_1h", "amt_ratio_30d", "fraude"]]
                     .rename(columns={"amt": "montant ($)", "fraude": "fraude réelle"}), height=250)
        rang = st.number_input("Rang de l'alerte à expliquer", 1, nb, 1)
        r = d.iloc[int(rang) - 1]
        non_nul = lambda v: None if pd.isna(v) else float(v)
        payload = {"amt": float(r["amt"]), "city_pop": int(r["city_pop"]), "distance_km": float(r["distance_km"]),
                   "heure": int(r["heure"]), "tx_1h": int(r["tx_1h"]), "tx_24h": int(r["tx_24h"]),
                   "amt_ratio_30d": non_nul(r["amt_ratio_30d"]), "secs_since_prev": non_nul(r["secs_since_prev"])}
        try:
            rep = requests.post(f"{api_url}/explain", json=payload, timeout=60).json()
        except Exception as e:
            st.error(f"API injoignable ({e}). Lance d'abord : python -m uvicorn main:app")
            st.stop()
        st.subheader(f"Score {rep['score']:.4f} · niveau : {rep['niveau']} · fraude réelle : {'oui' if r['fraude'] == 1 else 'non'}")
        g, dr = st.columns(2)
        with g:
            st.markdown("**Codes motifs**")
            for m in rep["codes_motifs"]:
                st.write(f"`{m['code']}` — {m['libelle']}")
        with dr:
            contrib = sorted(rep["contributions"], key=lambda x: x["contribution"])
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh([x["variable"] for x in contrib], [x["contribution"] for x in contrib],
                    color=["#d62728" if x["contribution"] > 0 else "#1f77b4" for x in contrib])
            ax.axvline(0, color="black", lw=.8)
            ax.set_xlabel("Contribution au score d'anomalie (rouge : plus suspect)")
            st.pyplot(fig)
