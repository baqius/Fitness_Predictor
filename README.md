# 🏋️ Fitness Predictor
### ML-Powered Personal Fitness Classification & Health Advisory System
---

## Project Overview

**Baqius Fitness Predictor** is an end-to-end machine learning solution that classifies whether an individual is physically fit based on their biometric and lifestyle data — and then delivers a personalised, evidence-based health advisory report, all from a clean, interactive web interface.

This project bridges the gap between raw health data and actionable insight. Rather than simply outputting a binary prediction, the system generates structured, clinically-grounded recommendations across five health domains: Immediate Priorities, Fitness & Exercise, Nutrition & Diet, Recovery & Lifestyle, and Strengths to Maintain.

## Business Problem & Motivation

Fitness apps, health insurers, corporate wellness programmes, and preventive healthcare platforms face a common challenge: **how do you assess a person's fitness level at scale without expensive clinical assessments?**

Traditional fitness evaluations require lab tests, gym equipment, or certified professionals — all of which create barriers in cost, time, and accessibility. Meanwhile, vast amounts of self-reported and wearable-derived health data sit underutilised.

**The goal of this project is to:**

- Build a machine learning classifier that predicts whether a person is fit (`is_fit = 1`) or not (`is_fit = 0`) based on 10 biometric and lifestyle features.
- Identify the key predictors of fitness to inform product design and health intervention strategy.
- Deploy the model as a user-facing web application that not only predicts fitness status but explains the reasoning and provides tailored, actionable health guidance.

### Why the Business Stakes Are High

In health classification problems, not all errors are equal:

- A **false negative** — predicting someone is fit when they are not — risks missing individuals who need health interventions. This is the higher-risk error, particularly if the model is used in insurance underwriting or preventive health screening.
- A **false positive** — predicting someone is unfit when they are fit — is less dangerous but wastes resources and may cause unnecessary concern.

This asymmetry directly shaped the modelling strategy: the project explicitly optimises for **recall on the unfit class** and recommends deploying with a lowered classification threshold (0.4 instead of the default 0.5) to reduce missed at-risk individuals.

---
## Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle](https://www.kaggle.com) |
| **Rows** | 2,000 |
| **Features** | 10 input features + 1 target |
| **Target** | `is_fit` (binary: 0 = Not Fit, 1 = Fit) |
| **Missing Values** | `sleep_hours` — 160 missing values (8%), handled via median imputation |
| **Class Balance** | Reasonably balanced (~60% Fit / 40% Not Fit) |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `age` | Integer | Age in years (18–79) |
| `height_cm` | Integer | Height in centimetres (150–199) |
| `weight_kg` | Integer | Weight in kilograms (30–250) |
| `heart_rate` | Float | Resting heart rate in bpm (45–110) |
| `blood_pressure` | Float | Systolic blood pressure in mmHg (80–180) |
| `sleep_hours` | Float | Average nightly sleep in hours (3–10) |
| `nutrition_quality` | Float | Self-reported nutrition quality score (1–10) |
| `activity_index` | Float | Physical activity level score (0–10) |
| `smokes` | Categorical | Smoking status (`yes` / `no` / `0`) |
| `gender` | Categorical | Gender (`M` / `F`) |
| `is_fit` | Integer | **Target variable** (0 = Not Fit, 1 = Fit) |

**Note:** The raw dataset contains inconsistent encoding in the `smokes` column (a mix of `"yes"`, `"no"`, and `0`). This messy, real-world issue is handled explicitly during the preprocessing stage.

---

## Exploratory Data Analysis

The EDA phase uncovered several important patterns before any modelling began:

**Weight is the strongest single visual discriminator.** The distribution of `weight_kg` shows meaningful separation between fit and unfit individuals, with unfit individuals skewing toward higher weights — consistent with the role BMI plays in clinical fitness assessment.

**Smoking status correlates with lower fitness.** Smokers appear disproportionately in the unfit class, confirming the well-established link between tobacco use and reduced cardiorespiratory fitness.

**Class balance is reasonable.** With approximately 60% fit and 40% not fit, the dataset does not require aggressive resampling (e.g. SMOTE), though threshold tuning remains important.

**Feature overlap is significant.** Several features (heart rate, blood pressure, sleep hours) have overlapping distributions across the two classes. This means no single feature is a perfect classifier — a key motivation for using ensemble methods over simple threshold rules.

**Baseline accuracy is 60.1%.** This is the majority-class baseline (always predicting Fit). Any model that does not significantly exceed this number offers no real value.

---

## Machine Learning Pipeline

### Preprocessing

All models share a unified scikit-learn `Pipeline` with a `ColumnTransformer`:

- **Numeric features** (`age`, `height_cm`, `weight_kg`, `heart_rate`, `blood_pressure`, `sleep_hours`, `nutrition_quality`, `activity_index`): Median imputation for missing values → `StandardScaler` for normalisation.
- **Categorical features** (`smokes`, `gender`): `OneHotEncoder` with `use_cat_names=True` to handle inconsistent encoding.

Wrapping preprocessing in a pipeline ensures there is **zero data leakage** between train and test sets — a common but serious error in data science work that this project handles correctly.

### Models Trained

Five classifiers were trained and compared:

| Model | Strength | Key Weakness |
|---|---|---|
| Logistic Regression | Fast, interpretable, generalises well | Assumes linear decision boundary |
| Decision Tree | Intuitive, no scaling required | Prone to overfitting |
| Random Forest | Robust, handles noise, strong out-of-the-box | Computationally heavier, less interpretable |
| Gradient Boosting | High accuracy potential, sequential correction | Sensitive to hyperparameters |
| XGBoost | State-of-the-art boosting, regularised | Risk of overfitting without careful tuning |

### Hyperparameter Tuning

All models were tuned using `GridSearchCV` with 5-fold stratified cross-validation, ensuring that class proportions are maintained in every fold.

---

## Model Results & Comparison

### Performance Summary

| Model | Train Accuracy | Test Accuracy | ROC-AUC | CV Accuracy (±Std) |
|---|---|---|---|---|
| **Logistic Regression** | 79.69% | **77.25%** | **0.8458** | 79.19% ± 0.018 |
| Decision Tree | 79.44% | 69.50% | 0.7352 | 72.25% ± 0.012 |
| Random Forest | 100.00% | 77.25% | 0.8178 | 78.38% ± 0.018 |
| Gradient Boosting | 84.88% | 76.50% | 0.8260 | 77.62% ± 0.019 |
| XGBoost | 86.50% | 75.75% | 0.8336 | 78.06% ± 0.019 |

### Selected Model: Logistic Regression

**Logistic Regression was selected as the best model.** While several ensemble methods achieved comparable test accuracy, Logistic Regression offered the best combination of:

- **Highest ROC-AUC (0.8458)** — the most important metric for a binary health classifier, as it measures discrimination ability independent of threshold.
- **No overfitting** — unlike Random Forest (100% train accuracy vs. 77.25% test), Logistic Regression generalised cleanly.
- **Interpretability** — coefficients can be inspected and explained to business stakeholders and clinicians.
- **Deployment simplicity** — fast inference, small model footprint, no risk of train-test discrepancy from tree memorisation.

### Classification Report (Best Model — Logistic Regression)

```
              precision    recall  f1-score   support

     Not Fit       0.78      0.86      0.82       240
         Fit       0.75      0.64      0.69       160

    accuracy                           0.77       400
   macro avg       0.77      0.75      0.76       400
weighted avg       0.77      0.77      0.77       400

ROC-AUC Score: 0.8458
```

The model achieves **86% recall on the Not Fit class** — meaning it correctly identifies 86 out of every 100 unfit individuals, which is critical for health applications where missing at-risk people carries the highest consequence.

---

## SHAP Explainability

The project integrates **SHAP (SHapley Additive exPlanations)** to go beyond aggregate feature importance and explain individual predictions.

SHAP provides:
- **Global feature importance** — which features, on average, drive the model's decisions most.
- **Individual prediction breakdown** — exactly how each feature pushed a specific person's prediction toward Fit or Not Fit.
- **Trust and auditability** — a critical requirement in health and insurance applications where regulators increasingly demand explainable AI.

Key SHAP findings aligned with clinical expectations: weight-derived features (directly linked to BMI) and `activity_index` emerged as the dominant contributors, while `smokes` and `sleep_hours` played secondary but meaningful roles.

---

## Business Conclusions & Risk Assessment

### Key Findings

1. **Weight and physical activity are the primary predictors.** This aligns with clinical evidence and validates that the model is learning real signal, not noise.

2. **Lifestyle factors matter independently.** Smoking status contributes meaningful predictive power beyond pure biometrics, confirming that fitness is a multidimensional health construct.

3. **Non-linearity is present but modest.** Ensemble models outperformed simpler models, but Logistic Regression's strong ROC-AUC suggests the problem is substantially linear — meaning the relationships between features and fitness are relatively direct.

4. **The model meaningfully outperforms the 60.1% baseline** across all five models tested, confirming genuine predictive value.

### Risk Assessment

| Error Type | Definition | Risk Level | Mitigation |
|---|---|---|---|
| False Negative | Predicted Fit, actually Not Fit | **High** — missed health intervention | Lower decision threshold to 0.4 |
| False Positive | Predicted Not Fit, actually Fit | **Low** — unnecessary concern or resources | Accept as acceptable cost |

---

### Features

**Input Panel**
- Age, height, weight, gender, smoking status (Personal Info column)
- Resting heart rate, systolic blood pressure, sleep hours, nutrition quality score, activity index (Health Metrics column)
- Real-time BMI calculation displayed with colour-coded classification (Underweight / Normal / Overweight / Obese)

**Prediction Output**
- Bold, colour-coded result banner: Green ✅ for Fit, Red ⚠️ for Not Fit
- Probability scores with visual progress bars for both classes
- Full confidence percentages to communicate uncertainty

**Personalised Health Report**
The tip engine (`build_tips`) generates a structured, prioritised report with colour-coded cards across five sections:

| Section | Purpose |
|---|---|
| 🚨 Immediate Priorities | High-urgency findings (obesity-range BMI, hypertension, heavy smoking) |
| 🏋️ Fitness & Exercise | Tailored exercise guidance based on activity index and age |
| 🥗 Nutrition & Diet | Evidence-based dietary recommendations based on nutrition score and BMI |
| 🌙 Recovery & Lifestyle | Sleep quality, stress, and recovery optimisation |
| 🌟 Strengths to Maintain | Positive reinforcement and optimisation for already-fit users |

Each tip card includes an **evidence-based explanation** of the health issue and a **concrete action step** for the user to take.
---

## Project Structure

```
baqius-fitness-predictor/
│
├── baqius_fitness.ipynb     # Full ML research notebook
├── app.py                   # Streamlit web application
├── fitness_model.pkl        # Serialised trained model (generated by notebook)
├── fitness_dataset.csv      # Raw dataset
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/baqius-fitness-predictor.git
cd baqius-fitness-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
streamlit
numpy
pandas
scikit-learn
category_encoders
xgboost
shap
matplotlib
seaborn
```

### Step 3: Train the Model (Run the Notebook)
Open and run `baqius_fitness.ipynb` in Jupyter to generate `fitness_model.pkl`. Alternatively, place your dataset at the path specified in the notebook.

### Step 4: Update the Model Path
In `app.py`, update line 58:
```python
MODEL_PATH = r"path/to/your/fitness_model.pkl"
```

### Step 5: Launch the App
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Data Manipulation** | pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Explainability** | SHAP |
| **Encoding** | category_encoders |
| **Visualisation** | Matplotlib, Seaborn |
| **Web Application** | Streamlit |
| **Model Serialisation** | pickle |

---

## Medical Disclaimer

> **This tool is for informational and educational purposes only and does not constitute medical advice.** The fitness classification and health recommendations produced by this application are generated by a machine learning model trained on a general dataset and should not be interpreted as a clinical diagnosis or personalised medical guidance. Always consult a qualified healthcare professional before making significant changes to your diet, exercise programme, medication, or lifestyle.
