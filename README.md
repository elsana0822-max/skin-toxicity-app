# 🧪 Skin Toxicity Predictor

A machine learning-based web application for predicting **skin irritation** and **skin sensitization** potential from molecular SMILES input.

Built as a final project for a computational toxicology course using RDKit, scikit-learn, and Flet.

---

## Live Demo

🔗 https://skin-toxicity-predictor.onrender.com

*(First load may take ~30 seconds on the free tier)*

---

## Overview

This app predicts three skin toxicity endpoints for any drug-like organic molecule:

| Endpoint | Assay Basis | Algorithm |
|---|---|---|
| Skin Irritation (qualitative) | In vivo (Draize / OECD TG 404) | Logistic Regression |
| Skin Sensitization (in vivo) | LLNA / GPMT (OECD TG 429/406) | Random Forest |
| Skin Sensitization (in vitro) | ARE-Nrf2 / h-CLAT (OECD TG 442D/E) | Logistic Regression |

---

## Model Pipeline

Each model is a scikit-learn `Pipeline` consisting of:

```
SimpleImputer (median) → StandardScaler → SelectKBest (ANOVA F) → Classifier
```

- **Descriptors:** 217 RDKit 2D molecular descriptors computed from SMILES
- **Feature selection:** Top-k features by ANOVA F-statistic
- **Hyperparameter tuning:** 5-fold stratified cross-validation (GridSearchCV)

---

## Performance

### Skin Irritation — Logistic Regression (k=50)

| Metric | Score |
|---|---|
| CV F1 | 0.552 |
| Test Accuracy | 0.900 |
| Test Precision | 1.000 |
| Test Recall | 0.667 |
| Test F1 | 0.800 |
| Test ROC-AUC | 0.905 |

### Skin Sensitization In Vivo — Random Forest (k=20)

| Metric | Score |
|---|---|
| CV F1 | 0.551 |
| Test Accuracy | 0.922 |
| Test Precision | 0.833 |
| Test Recall | 0.417 |
| Test F1 | 0.556 |
| Test ROC-AUC | 0.863 |

### Skin Sensitization In Vitro — Logistic Regression (k=20)

| Metric | Score |
|---|---|
| CV F1 | 0.882 |
| Test Accuracy | 0.800 |
| Test Precision | 0.789 |
| Test Recall | 1.000 |
| Test F1 | 0.882 |
| Test ROC-AUC | 0.547 |

---

## Features

- 🔬 Single SMILES prediction with probability scores
- 🖼️ 2D molecular structure visualization
- 📋 Example molecules (Ethanol, Acetic acid, Benzene, Triethylamine)
- 🌐 Web-based interface — no installation required for end users

---

## Local Installation

```bash
git clone https://github.com/elsana0822-max/skin-toxicity-app.git
cd skin-toxicity-app
pip install -r requirements.txt
python app_web.py
```

Then open `http://localhost:8080` in your browser.

**Requirements:** Python 3.10+, RDKit, scikit-learn 1.8.0

---

## Project Structure

```
skin-toxicity-app/
├── app_web.py                                       # Main web application
├── app_simple.py                                    # Desktop version
├── requirements.txt
├── render.yaml                                      # Render.com deployment config
├── model_best_skin_irritation_qualitative.joblib
├── model_best_skin_sensitization_invivo_call.joblib
├── model_best_skin_sensitization_invitro_call.joblib
└── example_batch.csv                                # Sample input file
```

---

## Tech Stack

| Component | Library |
|---|---|
| UI Framework | [Flet](https://flet.dev) 0.85 |
| Cheminformatics | [RDKit](https://www.rdkit.org) |
| Machine Learning | [scikit-learn](https://scikit-learn.org) 1.8 |
| Deployment | [Render.com](https://render.com) |

---

## QMRF Documentation

QSAR Model Reporting Format (QMRF) documents following the ECHA QMRF v2.1 standard are available for each model, covering endpoint definition, applicability domain, validation statistics, and mechanistic interpretation (OECD 5 Principles).

---

## Disclaimer

This tool is intended for **research and educational purposes only**. Predictions should not be used as the sole basis for regulatory decisions. Always validate results against experimental data and consult relevant OECD test guidelines.

---

## Author

**김서율** · 20251269  
Computational Toxicology, 2025
