# 🧪 Skin Toxicity Predictor

A machine learning-based web application for predicting skin irritation and skin sensitization potential from molecular SMILES input.

SMILES 입력 기반 피부 독성(자극성 / 감작성) 예측 머신러닝 웹 애플리케이션입니다.

컴퓨터 독성학 기말 프로젝트 | Final Project for Computational Toxicology Course (2025)

---

## 🔗 Live Demo

https://skin-toxicity-predictor.onrender.com

*(첫 접속 시 약 30초 소요될 수 있습니다 / First load may take ~30 seconds on the free tier)*

---

## 📌 Overview / 개요

This app predicts three skin toxicity endpoints for any drug-like organic molecule.  
약물 유사 유기 분자에 대해 아래 3가지 피부 독성 endpoint를 동시에 예측합니다.

| Endpoint | 예측 대상 | Assay Basis | Algorithm |
|---|---|---|---|
| Skin Irritation (qualitative) | 피부 자극성 | In vivo (Draize / OECD TG 404) | Logistic Regression |
| Skin Sensitization (in vivo) | 피부 감작성 (생체 내) | LLNA / GPMT (OECD TG 429/406) | Random Forest |
| Skin Sensitization (in vitro) | 피부 감작성 (시험관) | ARE-Nrf2 / h-CLAT (OECD TG 442D/E) | Logistic Regression |

---

## ⚙️ Model Pipeline / 모델 파이프라인

```
SimpleImputer (median) → StandardScaler → SelectKBest (ANOVA F) → Classifier
```

- **Descriptors / 기술자:** RDKit 2D molecular descriptors 217개
- **Feature selection / 특성 선택:** ANOVA F-통계량 기반 SelectKBest
- **Hyperparameter tuning / 하이퍼파라미터 최적화:** 5-fold Stratified CV + GridSearchCV

---

## 📊 Performance / 모델 성능

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

## ✨ Features / 주요 기능

- 🔬 SMILES 입력 → 3가지 endpoint 동시 예측 (확률값 포함)
- 🖼️ 2D 분자 구조 자동 시각화
- 📋 예시 분자 버튼 (Ethanol, Acetic acid, Benzene, Triethylamine)
- 🌐 웹 기반 — 사용자는 별도 설치 없이 브라우저에서 바로 사용 가능

---

## 💻 Local Installation / 로컬 설치

```bash
git clone https://github.com/elsana0822-max/skin-toxicity-app.git
cd skin-toxicity-app
pip install -r requirements.txt
python app_web.py
```

브라우저에서 `http://localhost:8080` 접속  
Open `http://localhost:8080` in your browser.

**Requirements:** Python 3.10+, RDKit, scikit-learn 1.8.0

---

## 📁 Project Structure / 파일 구조

```
skin-toxicity-app/
├── app_web.py                                       # 웹 배포용 메인 앱 / Main web app
├── app_simple.py                                    # 데스크톱 버전 / Desktop version
├── requirements.txt
├── render.yaml                                      # Render.com 배포 설정
├── model_best_skin_irritation_qualitative.joblib    # 피부 자극성 모델
├── model_best_skin_sensitization_invivo_call.joblib # 감작성 in vivo 모델
├── model_best_skin_sensitization_invitro_call.joblib# 감작성 in vitro 모델
└── example_batch.csv                                # 배치 처리 예시 파일
```

---

## 🛠 Tech Stack

| Component | Library |
|---|---|
| UI Framework | [Flet](https://flet.dev) 0.85 |
| Cheminformatics | [RDKit](https://www.rdkit.org) |
| Machine Learning | [scikit-learn](https://scikit-learn.org) 1.8 |
| Deployment | [Render.com](https://render.com) |

---

## 📄 QMRF Documentation

ECHA QMRF v2.1 표준을 따르는 QSAR 모델 보고서(QMRF)가 각 모델별로 작성되어 있습니다.  
Endpoint 정의, 적용 가능 영역(AD), 검증 통계, 기계론적 해석(OECD 5원칙) 포함.

---

## ⚠️ Disclaimer / 면책 조항

본 도구는 **연구 및 교육 목적**으로만 사용되어야 합니다. 예측 결과는 규제 의사결정의 단독 근거로 사용될 수 없으며, 반드시 실험 데이터 및 관련 OECD 시험 가이드라인과 함께 검토되어야 합니다.

This tool is intended for **research and educational purposes only**. Predictions should not be used as the sole basis for regulatory decisions.

