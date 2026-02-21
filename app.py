import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders import OneHotEncoder

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fitness Predictor",
    page_icon="🏋️",
    layout="wide"
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f7f9fc; }
    .result-fit {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white; border-radius: 16px; padding: 28px;
        text-align: center; font-size: 1.6rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(46,204,113,0.4); margin-bottom: 12px;
    }
    .result-notfit {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; border-radius: 16px; padding: 28px;
        text-align: center; font-size: 1.6rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(231,76,60,0.4); margin-bottom: 12px;
    }
    .tip-card {
        border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
        border-left: 5px solid; background: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .tip-card.warning { border-color: #e67e22; }
    .tip-card.danger  { border-color: #e74c3c; }
    .tip-card.info    { border-color: #3498db; }
    .tip-card.success { border-color: #2ecc71; }
    .tip-header { font-size: 1.05rem; font-weight: 700; margin-bottom: 8px; color: #2c3e50; }
    .tip-body   { font-size: 0.92rem; color: #4a4a4a; line-height: 1.70; margin-bottom: 10px; }
    .tip-action { font-size: 0.88rem; font-weight: 600; color: #2980b9;
                  border-top: 1px solid #eee; padding-top: 8px; }
    .section-label {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #888; margin: 22px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Model ────────────────────────────────────────────────────────────────────
# ✅ After — works on any machine, any OS
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitness_model.pkl")

# These are the EXACT 10 features the model was trained on (from the dataset).

NUMERIC_FEATURES     = ["age", "height_cm", "weight_kg", "heart_rate",
                         "blood_pressure", "sleep_hours", "nutrition_quality", "activity_index"]
CATEGORICAL_FEATURES = ["smokes", "gender"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_resource
def load_or_train_model(model_path: str):
    """Load saved model if available, otherwise train a fallback on synthetic data."""
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f), True

    # ── Fallback: synthetic data with the correct 10-feature schema ────────────
    np.random.seed(42)
    n = 2000

    age               = np.random.randint(18, 65, n)
    height_cm         = np.random.randint(150, 200, n)
    weight_kg         = np.random.randint(45, 120, n)
    heart_rate        = np.random.normal(72, 12, n).clip(45, 110)
    blood_pressure    = np.random.normal(120, 15, n).clip(80, 180)
    sleep_hours       = np.random.normal(7, 1.2, n).clip(3, 10)
    nutrition_quality = np.random.uniform(1, 10, n)   # score 1–10
    activity_index    = np.random.uniform(0, 10, n)   # score 0–10
    smokes            = np.random.choice(["yes", "no"], n, p=[0.25, 0.75])
    gender            = np.random.choice(["male", "female"], n)

    bmi = weight_kg / (height_cm / 100) ** 2
    score = (
        (activity_index > 5).astype(int) * 2 +
        (bmi < 25).astype(int) * 2 +
        (sleep_hours >= 7).astype(int) +
        (nutrition_quality > 6).astype(int) +
        (smokes == "no").astype(int) -
        (blood_pressure > 140).astype(int)
    )
    is_fit = (score >= 4).astype(int)

    df = pd.DataFrame({
        "age": age, "height_cm": height_cm, "weight_kg": weight_kg,
        "heart_rate": heart_rate, "blood_pressure": blood_pressure,
        "sleep_hours": sleep_hours, "nutrition_quality": nutrition_quality,
        "activity_index": activity_index, "smokes": smokes, "gender": gender
    })

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(use_cat_names=True))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    pipe.fit(df, pd.Series(is_fit))
    return pipe, False


model, model_loaded = load_or_train_model(MODEL_PATH)


# ─── Tip Renderer ─────────────────────────────────────────────────────────────
def tip_card(icon, header, body, action, card_type="info"):
    st.markdown(f"""
    <div class="tip-card {card_type}">
        <div class="tip-header">{icon} {header}</div>
        <div class="tip-body">{body}</div>
        <div class="tip-action">✅ Action Step: {action}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Tip Logic ────────────────────────────────────────────────────────────────
def build_tips(age, height_cm, weight_kg, heart_rate, blood_pressure,
               sleep_hours, nutrition_quality, activity_index, smokes, gender, pred):

    bmi = weight_kg / (height_cm / 100) ** 2

    tips = {
        "Immediate Priorities":  [],
        "Fitness & Exercise":    [],
        "Nutrition & Diet":      [],
        "Recovery & Lifestyle":  [],
        "Strengths to Maintain": []
    }

    # ── BMI (derived) ─────────────────────────────────────────────────────────
    if bmi >= 30:
        tips["Immediate Priorities"].append((
            "danger", "⚠️", "Obesity-Range BMI Detected",
            f"Your calculated BMI of <b>{bmi:.1f}</b> (from your height and weight) places you in the "
            "<b>obese category (≥ 30)</b>. Obesity is a well-established risk factor for type 2 diabetes, "
            "hypertension, coronary artery disease, sleep apnoea, and certain cancers. It also places "
            "significant mechanical stress on the joints, accelerating cartilage wear. Even a modest "
            "5–10% reduction in body weight produces measurable improvements in blood pressure, "
            "blood glucose, cholesterol, and inflammatory markers.",
            "Begin with low-impact aerobic exercise (walking, swimming, cycling) 4–5× per week for 30 minutes. "
            "Work with a physician or registered dietitian to establish a safe caloric deficit of 400–600 kcal/day. "
            "Avoid crash dieting — gradual loss of 0.5–1 kg per week is safest and most maintainable."
        ))
    elif bmi >= 25:
        tips["Immediate Priorities"].append((
            "warning", "📊", "Overweight BMI — Proactive Change Recommended",
            f"Your calculated BMI of <b>{bmi:.1f}</b> falls in the <b>overweight range (25–29.9)</b>. "
            "Sustained excess weight raises long-term cardiovascular risk and accelerates decline in "
            "VO₂ max (aerobic capacity), one of the strongest predictors of longevity.",
            "Target a sustainable deficit of 200–300 kcal/day through food adjustments "
            "(reduce ultra-processed foods, increase vegetables and protein). "
            "Add 2–3 weekly cardio sessions of 30–45 minutes. Reassess in 8 weeks."
        ))
    elif bmi < 18.5:
        tips["Immediate Priorities"].append((
            "warning", "📉", "Underweight BMI — Nutritional Evaluation Needed",
            f"Your calculated BMI of <b>{bmi:.1f}</b> is below the healthy threshold of 18.5. "
            "Being underweight is associated with impaired immune function, reduced bone mineral density, "
            "muscle wasting, hormonal imbalances, and micronutrient deficiencies.",
            "Increase daily caloric intake by 300–500 kcal using nutrient-dense foods: lean proteins, "
            "whole grains, healthy fats (avocado, nuts, olive oil), and dairy. "
            "Aim for 3 balanced meals and 2 snacks per day."
        ))
    else:
        tips["Strengths to Maintain"].append((
            "success", "✅", "Healthy BMI — Excellent Foundation",
            f"Your calculated BMI of <b>{bmi:.1f}</b> falls in the <b>healthy range (18.5–24.9)</b>. "
            "This reduces your risk of weight-related chronic diseases and supports better cardiovascular "
            "function, improved metabolic health, and greater physical capacity.",
            "Continue your balanced approach. Monitor your weight weekly and maintain it through "
            "consistent exercise and avoiding prolonged periods of inactivity."
        ))

    # ── Heart Rate ────────────────────────────────────────────────────────────
    if heart_rate > 90:
        tips["Immediate Priorities"].append((
            "danger", "❤️", "Elevated Resting Heart Rate — Cardiovascular Concern",
            f"A resting heart rate of <b>{heart_rate:.0f} bpm</b> is above the normal threshold of 60–80 bpm. "
            "A chronically elevated resting HR signals cardiovascular inefficiency and is associated with "
            "hypertension, chronic stress, dehydration, thyroid dysfunction, and poor aerobic fitness. "
            "Large-scale studies confirm that resting HR above 90 bpm independently predicts cardiovascular "
            "events and premature mortality.",
            "Begin structured aerobic conditioning: 30 minutes of moderate-intensity cardio "
            "(brisk walking, cycling, swimming) 4–5× per week. Stay well-hydrated (2–3 litres/day). "
            "Limit caffeine and alcohol, both of which raise resting HR. "
            "If HR stays above 90 bpm after 4 weeks, seek a cardiovascular evaluation."
        ))
    elif heart_rate > 80:
        tips["Fitness & Exercise"].append((
            "warning", "❤️", "Above-Average Resting Heart Rate",
            f"A resting HR of <b>{heart_rate:.0f} bpm</b> is slightly elevated. "
            "Optimal cardiovascular health is associated with resting HR of 60–70 bpm; "
            "well-trained athletes typically achieve 40–60 bpm. "
            "Improving aerobic base fitness directly lowers resting HR over 6–12 weeks.",
            "Add 3–4 weekly Zone 2 cardio sessions (conversational pace, slightly breathless). "
            "This is the most effective training zone for improving stroke volume and cardiac efficiency."
        ))
    elif heart_rate <= 55:
        tips["Strengths to Maintain"].append((
            "success", "❤️", "Athletic Resting Heart Rate",
            f"A resting HR of <b>{heart_rate:.0f} bpm</b> reflects excellent cardiovascular conditioning — "
            "your heart pumps more blood per beat, requiring fewer beats to maintain circulation. "
            "This is a hallmark of aerobic fitness associated with longer life expectancy.",
            "Continue your cardiovascular training. If resting HR drops below 40 bpm with symptoms "
            "of dizziness or fatigue, consult a cardiologist to rule out bradyarrhythmia."
        ))
    else:
        tips["Strengths to Maintain"].append((
            "success", "❤️", "Healthy Resting Heart Rate",
            f"Your resting HR of <b>{heart_rate:.0f} bpm</b> is within the healthy normal range, "
            "indicating good cardiovascular function and efficient cardiac output.",
            "Maintain regular aerobic exercise to sustain or further improve cardiac efficiency. "
            "Track your resting HR each morning — a consistent rise of 5+ bpm can signal overtraining or illness."
        ))

    # ── Blood Pressure ────────────────────────────────────────────────────────
    if blood_pressure >= 140:
        tips["Immediate Priorities"].append((
            "danger", "🩺", "High Blood Pressure (Hypertension Stage 2)",
            f"A blood pressure reading of <b>{blood_pressure:.0f} mmHg</b> (systolic) is in the "
            "<b>hypertension stage 2 range (≥ 140)</b>. "
            "Hypertension dramatically increases the risk of stroke, heart attack, kidney disease, "
            "and heart failure. It is often called the 'silent killer' because it causes organ damage "
            "with no symptoms. Uncontrolled hypertension at this level requires medical evaluation.",
            "See a doctor promptly — medication may be required at this level. "
            "Simultaneously adopt the DASH diet (rich in fruits, vegetables, whole grains, low-fat dairy), "
            "reduce sodium intake to under 2,300 mg/day, limit alcohol, quit smoking, "
            "and begin moderate aerobic exercise 5× per week."
        ))
    elif blood_pressure >= 130:
        tips["Immediate Priorities"].append((
            "warning", "🩺", "Elevated Blood Pressure (Hypertension Stage 1)",
            f"Your blood pressure of <b>{blood_pressure:.0f} mmHg</b> (systolic) falls in the "
            "<b>hypertension stage 1 range (130–139)</b>. "
            "This significantly raises cardiovascular risk over time and often progresses without lifestyle changes.",
            "Reduce sodium intake, increase potassium-rich foods (bananas, leafy greens, potatoes), "
            "maintain a healthy weight, exercise regularly (150 min/week moderate intensity), "
            "limit alcohol, and manage stress actively. Recheck in 1 month."
        ))
    elif blood_pressure >= 120:
        tips["Fitness & Exercise"].append((
            "info", "🩺", "Elevated Blood Pressure — Monitor Closely",
            f"Your blood pressure of <b>{blood_pressure:.0f} mmHg</b> (systolic) is in the "
            "<b>elevated range (120–129)</b> — above optimal but not yet classified as hypertension. "
            "Without intervention, this frequently progresses to stage 1 hypertension within a few years.",
            "Adopt heart-healthy habits now: reduce sodium and processed foods, increase aerobic exercise, "
            "manage stress (meditation, breathing exercises), limit caffeine, and recheck blood pressure monthly."
        ))
    else:
        tips["Strengths to Maintain"].append((
            "success", "🩺", "Healthy Blood Pressure",
            f"Your blood pressure of <b>{blood_pressure:.0f} mmHg</b> (systolic) is in the "
            "<b>normal range (< 120)</b>. This significantly reduces your risk of cardiovascular events, "
            "stroke, and kidney disease.",
            "Continue your heart-healthy habits. Monitor blood pressure annually or more frequently "
            "if you have a family history of hypertension."
        ))

    # ── Sleep ─────────────────────────────────────────────────────────────────
    if sleep_hours < 6:
        tips["Recovery & Lifestyle"].append((
            "danger", "😴", "Severe Sleep Deprivation — Critical Health Risk",
            f"At only <b>{sleep_hours:.1f} hours of sleep per night</b>, you are in chronic sleep deprivation. "
            "This suppresses growth hormone secretion (muscle repair and fat metabolism); elevates cortisol, "
            "promoting abdominal fat storage; disrupts hunger hormones, increasing appetite for high-calorie foods; "
            "impairs insulin sensitivity, raising diabetes risk; and reduces physical performance and cognitive function. "
            "Chronic short sleep is independently associated with higher all-cause mortality.",
            "Establish a fixed sleep schedule — consistent bedtime and wake time, even on weekends. "
            "Create a dark, cool (18–20°C), quiet sleep environment. Avoid screens 60 min before bed. "
            "Limit caffeine after 1–2 pm. If 7 hours remains unachievable, consult a GP to rule out "
            "sleep apnoea or insomnia."
        ))
    elif sleep_hours < 7:
        tips["Recovery & Lifestyle"].append((
            "warning", "😴", "Slightly Insufficient Sleep",
            f"Averaging <b>{sleep_hours:.1f} hours/night</b> puts you just below the recommended 7–9 hours. "
            "Even mild sleep restriction impairs muscle protein synthesis, reduces next-day exercise capacity, "
            "and elevates inflammatory markers. Athletes sleeping under 7 hours sustain significantly more "
            "injuries than those sleeping 8+.",
            "Move your bedtime 20–30 minutes earlier this week. Avoid alcohol within 3 hours of sleep — "
            "it fragments sleep architecture and reduces restorative REM quality."
        ))
    elif sleep_hours > 9.5:
        tips["Recovery & Lifestyle"].append((
            "info", "😴", "Excess Sleep — Investigate Underlying Causes",
            f"Regularly sleeping <b>{sleep_hours:.1f} hours/night</b> may indicate underlying fatigue, "
            "depression, hypothyroidism, or sleep apnoea. Prolonged sleep beyond 9 hours is associated "
            "with increased inflammation and paradoxically higher cardiovascular risk.",
            "Focus on sleep quality over quantity. If you wake unrefreshed after 9+ hours, consult a GP. "
            "Increase daytime physical activity to regulate your circadian rhythm."
        ))
    else:
        tips["Strengths to Maintain"].append((
            "success", "😴", "Excellent Sleep Duration",
            f"Getting <b>{sleep_hours:.1f} hours of sleep per night</b> is optimal. "
            "Quality sleep drives muscle repair via growth hormone, consolidates motor learning from training, "
            "regulates hunger hormones, and maintains cognitive sharpness.",
            "Protect your sleep schedule as non-negotiable. Maintain the same sleep and wake time "
            "7 days a week for optimal circadian rhythm consistency."
        ))

    # ── Nutrition Quality ─────────────────────────────────────────────────────
    if nutrition_quality < 4:
        tips["Nutrition & Diet"].append((
            "danger", "🥗", "Poor Nutrition Quality — Immediate Dietary Overhaul Needed",
            f"A nutrition quality score of <b>{nutrition_quality:.1f}/10</b> indicates a significantly "
            "poor diet. Poor nutrition is a foundational barrier to fitness — it undermines energy levels, "
            "recovery, immune function, hormonal balance, and body composition regardless of how much you exercise. "
            "A low-quality diet high in ultra-processed foods, added sugars, and refined carbohydrates "
            "promotes chronic inflammation, insulin resistance, and visceral fat accumulation.",
            "Start by eliminating ultra-processed foods and sugary drinks. Replace them with whole foods: "
            "lean proteins (chicken, fish, legumes, eggs), complex carbohydrates (oats, brown rice, sweet potato), "
            "healthy fats (avocado, nuts, olive oil), and abundant vegetables. "
            "Aim for 5 portions of vegetables and fruit daily. Consider tracking meals for 2 weeks to build awareness."
        ))
    elif nutrition_quality < 6:
        tips["Nutrition & Diet"].append((
            "warning", "🥗", "Below-Average Nutrition Quality — Meaningful Improvement Needed",
            f"Your nutrition score of <b>{nutrition_quality:.1f}/10</b> suggests there is significant room "
            "for improvement. Suboptimal nutrition limits recovery between workouts, impairs fat metabolism, "
            "reduces energy availability, and blunts immune response. Even moderate dietary improvements "
            "produce measurable changes in body composition and performance within 4–8 weeks.",
            "Focus on three changes: (1) increase protein intake to 1.2–1.6 g/kg body weight, "
            "(2) add vegetables to at least 2 meals per day, "
            "(3) replace one processed food item per day with a whole-food alternative. "
            "Build from there gradually."
        ))
    elif nutrition_quality >= 8:
        tips["Strengths to Maintain"].append((
            "success", "🥗", "Excellent Nutrition Quality",
            f"A nutrition score of <b>{nutrition_quality:.1f}/10</b> reflects a high-quality diet. "
            "Good nutrition is the foundation of fitness — it fuels performance, accelerates recovery, "
            "supports hormonal balance, and optimises body composition. "
            "Consistent high-quality eating is one of the most powerful longevity tools available.",
            "Maintain your dietary habits. Periodically reassess micronutrient intake "
            "(Vitamin D, B12, iron, omega-3s) and consider annual blood work to ensure optimal levels."
        ))
    else:
        tips["Nutrition & Diet"].append((
            "info", "🥗", "Moderate Nutrition Quality — Room to Optimise",
            f"Your nutrition score of <b>{nutrition_quality:.1f}/10</b> is reasonable but has clear "
            "room for improvement. Optimising diet quality — even from moderate to good — "
            "can meaningfully improve energy, recovery speed, body composition, and long-term health.",
            "Target one nutritional upgrade per week: swap refined grains for whole grains, "
            "add a portion of oily fish twice weekly for omega-3s, increase fibre via legumes and vegetables, "
            "and reduce added sugar. Small, consistent changes compound over time."
        ))

    # ── Activity Index ────────────────────────────────────────────────────────
    if activity_index < 3:
        tips["Fitness & Exercise"].append((
            "danger", "🏃", "Very Low Activity Level — Urgent Priority",
            f"An activity index of <b>{activity_index:.1f}/10</b> indicates a largely sedentary lifestyle. "
            "Physical inactivity is among the top preventable causes of chronic disease and premature death. "
            "It accelerates muscle loss, cardiovascular deconditioning, metabolic decline, and worsens "
            "mental health. The WHO recommends at least 150 minutes of moderate aerobic activity per week.",
            "Begin with 20-minute walks 3× per week. After 2 weeks, add bodyweight exercises: "
            "squats, push-ups, lunges, and planks. Build up gradually to avoid injury. "
            "Find an activity you genuinely enjoy — adherence is the most critical factor."
        ))
    elif activity_index < 5:
        tips["Fitness & Exercise"].append((
            "warning", "🏃", "Below-Average Activity Level",
            f"Your activity index of <b>{activity_index:.1f}/10</b> is below the level needed for "
            "meaningful cardiovascular and muscular adaptation. Insufficient physical activity limits "
            "aerobic capacity, promotes fat accumulation, and accelerates age-related muscle loss.",
            "Add 1–2 structured exercise sessions per week to your current routine. "
            "Mix cardio (30-min brisk walk or cycle) with strength training (bodyweight or resistance). "
            "Aim to increase your activity score progressively over 4–6 weeks."
        ))
    elif activity_index >= 8:
        tips["Strengths to Maintain"].append((
            "success", "🏃", "High Activity Level — Excellent Work",
            f"An activity index of <b>{activity_index:.1f}/10</b> reflects a highly active lifestyle. "
            "Regular, sustained physical activity at this level significantly reduces all-cause mortality, "
            "improves insulin sensitivity, supports healthy body weight, and enhances mental wellbeing.",
            "Ensure your activity includes variety: cardiovascular, strength, and flexibility components. "
            "Monitor for overtraining — include at least 1–2 rest or active recovery days per week."
        ))
    else:
        tips["Strengths to Maintain"].append((
            "success", "🏃", "Good Activity Level",
            f"Your activity index of <b>{activity_index:.1f}/10</b> is solid and meets general health guidelines. "
            "Regular physical activity at this level supports cardiovascular health, metabolic function, "
            "and mental wellbeing.",
            "Continue your current activity habits. To progress further, apply progressive overload — "
            "gradually increase intensity, duration, or frequency every 2–3 weeks."
        ))

    # ── Smoking ───────────────────────────────────────────────────────────────
    if smokes == "yes":
        tips["Immediate Priorities"].append((
            "danger", "🚭", "Smoking — The Single Most Damaging Modifiable Risk Factor",
            "Smoking is the leading preventable cause of disease and premature death globally. "
            "From a fitness perspective it reduces VO₂ max by up to 15% by impairing oxygen delivery; "
            "causes carbon monoxide to bind haemoglobin, cutting blood oxygen-carrying capacity; "
            "promotes atherosclerosis limiting cardiovascular performance; impairs collagen synthesis "
            "slowing post-exercise tendon and ligament repair; and reduces bone mineral density raising fracture risk. "
            "No level of fitness or diet quality can fully compensate. "
            "Within 3 months of quitting, lung function and circulation measurably improve.",
            "Seek structured cessation support — the most effective approach combines Nicotine Replacement "
            "Therapy (patches, gum, spray) + prescription medication (varenicline/bupropion) + behavioural counselling. "
            "Set a firm quit date and enlist social support for accountability."
        ))
    else:
        tips["Strengths to Maintain"].append((
            "success", "🚭", "Non-Smoker — Significant Health Advantage",
            "Not smoking is one of the most impactful health decisions you can make. "
            "Non-smokers have significantly higher VO₂ max, better cardiovascular efficiency, "
            "lower cancer risk, improved lung function, and faster exercise recovery compared to smokers.",
            "Continue to avoid tobacco and passive smoke exposure. "
            "Be mindful that vaping and smokeless tobacco also carry significant health risks."
        ))

    # ── Age-Specific ──────────────────────────────────────────────────────────
    if age >= 60:
        tips["Fitness & Exercise"].append((
            "info", "🧓", "Older Adult: Strength, Balance & Bone Health are Critical",
            f"At <b>{age} years</b>, sarcopenia prevention (muscle loss ~1–2%/year without resistance training), "
            "bone density maintenance, and fall prevention become paramount. Falls are among the leading "
            "causes of hospitalisation in this age group. Regular exercise also reduces dementia risk by 30–40%.",
            "Incorporate resistance training 2–3× per week using chair squats, wall push-ups, and resistance bands. "
            "Add balance work: single-leg standing, heel-to-toe walking, yoga, or Tai Chi. "
            "Ensure adequate protein (1.2–1.6 g/kg), calcium (1,200 mg/day), and Vitamin D (800–1,000 IU/day)."
        ))
    elif age >= 50:
        tips["Fitness & Exercise"].append((
            "info", "🧓", "Middle Age: Invest in Preventive Fitness Now",
            f"At <b>{age} years</b>, hormonal shifts accelerate muscle loss, visceral fat redistribution, "
            "and bone density decline. Resistance training and adequate protein counteract these changes effectively.",
            "Prioritise strength training 2–3× per week with compound movements. "
            "Increase dietary protein to 1.2–1.6 g/kg body weight. "
            "Schedule regular health screenings: blood pressure, lipid panel, blood glucose, bone density."
        ))
    elif age <= 25:
        tips["Fitness & Exercise"].append((
            "info", "🌱", "Young Adult: Build Your Lifelong Foundation Now",
            f"At <b>{age} years</b>, your body is at peak adaptive capacity. Bone density peaks in your late 20s — "
            "this is the critical window for weight-bearing exercise and calcium-rich nutrition. "
            "Habits established now compound powerfully over decades.",
            "Build diverse exercise habits: strength + cardio + flexibility. "
            "Eat a nutrient-rich diet with adequate protein, calcium, and iron. "
            "Protect sleep — debt in your 20s disrupts hormonal development and long-term metabolic programming."
        ))

    # ── Closing ───────────────────────────────────────────────────────────────
    if pred == 1:
        tips["Strengths to Maintain"].append((
            "success", "🌟", "Overall Assessment: Maintain and Optimise",
            "Your profile reflects a generally fit individual. Fitness is not a fixed destination — "
            "it requires continual adjustment as your body adapts and life circumstances evolve. "
            "The most resilient outcomes come from continuously setting new goals and periodically reassessing.",
            "Schedule a comprehensive fitness assessment every 3–6 months: body composition, "
            "cardiovascular fitness test, strength benchmarks, and blood biomarkers "
            "(lipids, glucose, haemoglobin, Vitamin D)."
        ))
    else:
        tips["Immediate Priorities"].append((
            "info", "🌱", "Starting Point: Progress is Always Possible",
            "A 'not fit' classification is not a verdict — it is a baseline. "
            "The human body is remarkably adaptable at any age or fitness level. "
            "With consistent, evidence-informed changes over 8–12 weeks, measurable improvements "
            "in body composition, energy, cardiovascular fitness, and mental wellbeing are achievable for almost everyone.",
            "Choose ONE area from these recommendations to act on this week. "
            "Implement it consistently for 14 days before adding a second change. "
            "Track progress every 2 weeks — even small improvements compound into transformative results."
        ))

    return tips


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🏋️ Fitness Predictor")
st.markdown(
    "Enter your health details below to receive a personalised fitness prediction "
    "and a comprehensive, evidence-based assessment report."
)

if not model_loaded:
    st.warning(
        "⚠️ Saved model not found — a fallback model trained on synthetic data is being used. "
        "Update `MODEL_PATH` at the top of `fitness_app.py` to point to your real `fitness_model.pkl`."
    )

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Personal Info")
    age            = st.slider("Age (years)", 10, 80, 30)
    height_cm      = st.slider("Height (cm)", 140, 210, 170)
    weight_kg      = st.slider("Weight (kg)", 40, 150, 70)
    gender         = st.radio("Gender", ["male", "female"], horizontal=True)
    smokes         = st.radio("Do you smoke?", ["no", "yes"], horizontal=True)

with col2:
    st.subheader("🏥 Health Metrics")
    heart_rate        = st.slider("Resting Heart Rate (bpm)", 40, 120, 72)
    blood_pressure    = st.slider("Systolic Blood Pressure (mmHg)", 80, 200, 120)
    sleep_hours       = st.slider("Sleep (hours/night)", 3.0, 12.0, 7.0, 0.5)
    nutrition_quality = st.slider("Nutrition Quality Score (1–10)", 1.0, 10.0, 6.0, 0.5)
    activity_index    = st.slider("Activity Index (0–10)", 0.0, 10.0, 5.0, 0.5)

# Derived BMI display
bmi = weight_kg / (height_cm / 100) ** 2
bmi_label = ("Underweight" if bmi < 18.5 else
             "Normal weight" if bmi < 25 else
             "Overweight" if bmi < 30 else "Obese")
bmi_icons = {"Underweight": "🟡", "Normal weight": "🟢", "Overweight": "🟠", "Obese": "🔴"}
st.markdown(f"**Calculated BMI:** `{bmi:.1f}` &nbsp;&nbsp; {bmi_icons[bmi_label]} *{bmi_label}*")

st.divider()

if st.button("🔍 Predict My Fitness", use_container_width=True, type="primary"):

    # Build input DataFrame with EXACTLY the 10 features the model expects
    # Note: smokes is encoded as int (0/1) to match training preprocessing
    input_data = pd.DataFrame([{
        "age":               age,
        "height_cm":         height_cm,
        "weight_kg":         weight_kg,
        "heart_rate":        heart_rate,
        "blood_pressure":    blood_pressure,
        "sleep_hours":       sleep_hours,
        "nutrition_quality": nutrition_quality,
        "activity_index":    activity_index,
        "smokes":            1 if smokes == "yes" else 0,
        "gender":            gender
    }])

    try:
        pred  = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    fit_prob    = proba[1] * 100
    notfit_prob = proba[0] * 100

    # ── Result Banner ──────────────────────────────────────────────────────
    st.subheader("📊 Assessment Result")
    if pred == 1:
        st.markdown(
            '<div class="result-fit">✅ You are FIT! Keep it up! 💪</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-notfit">⚠️ Not Fit Yet — But you can change that! 🌱</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🟢 Fit Probability", f"{fit_prob:.1f}%")
        st.progress(fit_prob / 100)
    with c2:
        st.metric("🔴 Not Fit Probability", f"{notfit_prob:.1f}%")
        st.progress(notfit_prob / 100)

    # ── Recommendations ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Personalised Health & Fitness Recommendations")
    st.markdown(
        "Recommendations below are tailored to your inputs, organised by category, "
        "and prioritised by clinical urgency. Each includes evidence-based context and a concrete action step."
    )

    all_tips = build_tips(
        age, height_cm, weight_kg, heart_rate, blood_pressure,
        sleep_hours, nutrition_quality, activity_index, smokes, gender, pred
    )

    SECTION_ORDER = [
        "Immediate Priorities", "Fitness & Exercise",
        "Nutrition & Diet", "Recovery & Lifestyle", "Strengths to Maintain"
    ]
    SECTION_ICONS = {
        "Immediate Priorities":  "🚨",
        "Fitness & Exercise":    "🏋️",
        "Nutrition & Diet":      "🥗",
        "Recovery & Lifestyle":  "🌙",
        "Strengths to Maintain": "🌟"
    }

    for section in SECTION_ORDER:
        items = all_tips.get(section, [])
        if items:
            st.markdown(
                f'<div class="section-label">{SECTION_ICONS[section]} {section}</div>',
                unsafe_allow_html=True
            )
            for (card_type, icon, header, body, action) in items:
                tip_card(icon, header, body, action, card_type)

    st.divider()
    st.markdown(
        "**⚠️ Medical Disclaimer:** This tool is for informational and educational purposes only "
        "and does not constitute medical advice. Always consult a qualified healthcare professional "
        "before making significant changes to your diet, exercise programme, or lifestyle."
    )

