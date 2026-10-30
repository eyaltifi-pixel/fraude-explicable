# Détection de fraude explicable (Challenge 2, Data Mining, ENSI)

Application web de détection de fraude sur cartes bancaires, avec **détecteurs non supervisés**
(Isolation Forest retenu), seuil d'alerte fixé par un **coût métier**, explications **SHAP** et **API FastAPI**.

Jeu de données : *Credit Card Transactions Fraud Detection* (simulateur Sparkov, Kaggle `kartik2112/fraud-detection`).

## Contenu
| Fichier | Rôle |
|---|---|
| `Projet_Data_Mining_challenge_2.ipynb` | Modules 1 à 6 : données, variables, modèles, évaluation et seuil, XAI, API |
| `api/main.py` | API FastAPI (`/predict`, `/predict/batch`, `/explain`, `/model-info`, `/health`) |
| `api/explication.py` | Regroupement des contributions SHAP et codes motifs |
| `api/artefacts.joblib` | Modèle, imputer, scaler, seuils (généré par le notebook) |
| `api/app.py` | Application Streamlit (curseurs de seuil et de budget, explication d'une alerte) |
| `api/donnees_app.csv.gz` | Scores et variables des jeux validation et test (généré par le notebook) |

## Lancer le projet
```bash
pip install -r requirements.txt
cd api
python -m uvicorn main:app --port 8001        # terminal 1 : API  (docs : http://127.0.0.1:8001/docs)
streamlit run app.py                           # terminal 2 : application
```
Dans l'application, renseigner l'URL de l'API (`http://127.0.0.1:8001`).

## Méthode en bref
- Séparation **chronologique** 60 / 20 / 20, variables comportementales calculées sur le passé de la carte uniquement.
- Détecteurs non supervisés comparés en validation : Isolation Forest, LOF, One-Class SVM.
- Seuil choisi en validation en minimisant : montants des fraudes manquées + coût d'analyse des alertes, puis confirmé sur le test.
- SHAP explique le **score d'anomalie** (pourquoi une transaction est atypique), pas une probabilité de fraude.

## Auteure
Eya Ltifi, ENSI, II3-IA.
