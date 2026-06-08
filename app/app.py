import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "data"

# Load model dan scaler
model = joblib.load(MODEL_DIR / "linear_regression.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>

/* =========================
   GLOBAL
========================= */

.block-container{
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* =========================
   HERO CARD
========================= */

.hero-card{

    background:
    linear-gradient(
        135deg,
        #2563eb 0%,
        #7c3aed 50%,
        #9333ea 100%
    );

    padding: 3rem;

    border-radius: 28px;

    text-align: center;

    box-shadow:
    0 20px 50px rgba(99,102,241,0.35);

    margin-bottom: 2rem;

    position: relative;

    overflow: hidden;
}

.hero-card::before{

    content: "";

    position: absolute;

    width: 300px;
    height: 300px;

    background:
    radial-gradient(
        rgba(255,255,255,0.25),
        transparent
    );

    top: -150px;
    right: -100px;
}

.hero-title{

    color: white;

    font-size: 3rem;

    font-weight: 800;

    margin-bottom: 0.5rem;
}

.hero-subtitle{

    color: rgba(255,255,255,0.9);

    font-size: 1rem;
}

/* =========================
   INPUTS
========================= */

[data-baseweb="input"]{

    border-radius: 16px !important;
}

.stSelectbox > div > div{

    border-radius: 16px !important;
}

/* =========================
   METRIC CARD
========================= */

[data-testid="metric-container"]{

    backdrop-filter: blur(20px);

    border-radius: 20px;

    padding: 1rem;

    border: 1px solid rgba(120,120,120,0.15);

    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover{

    transform: translateY(-3px);

    box-shadow:
    0 12px 25px rgba(0,0,0,0.12);
}

/* =========================
   BUTTON
========================= */

.stButton > button{

    width: 100%;

    height: 58px;

    border: none;

    border-radius: 18px;

    font-size: 18px;

    font-weight: 700;

    color: white;

    background:
    linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    transition: all 0.3s ease;

    box-shadow:
    0 10px 25px rgba(99,102,241,0.35);
}

.stButton > button:hover{

    transform: translateY(-2px);

    box-shadow:
    0 15px 35px rgba(99,102,241,0.45);
}

/* =========================
   SIDEBAR CARD
========================= */

.sidebar-card{

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    padding: 1.5rem;

    border-radius: 20px;

    color: white;

    box-shadow:
    0 10px 25px rgba(99,102,241,0.25);
}

/* =========================
   EXPANDER
========================= */

.streamlit-expanderHeader{

    font-weight: 700;
}

/* =========================
   SCROLLBAR
========================= */

::-webkit-scrollbar{
    width: 10px;
}

::-webkit-scrollbar-thumb{

    background:
    linear-gradient(
        #2563eb,
        #7c3aed
    );

    border-radius: 999px;
}

</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
# Analytics Dashboard
""")
st.markdown("""
<div class="hero-card">
    <h1 class="hero-title">Student Performance Prediction</h1>
    <p class="hero-subtitle">Predicting exam scores based on academic, social, and demographic factors.</p>

<h3>Model Overview</h3>

<b>Best Model</b><br>
Linear Regression<br><br>

<b>Accuracy (R²)</b><br>
76.96%<br><br>

<b>Dataset</b><br>
6607 Students<br><br>

<b>Features</b><br>
27 Variables

</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("About This Model"):
    st.write("""
This machine learning model predicts
student exam scores based on
academic, social, and demographic
factors.

Model:
Linear Regression

Performance:
R² = 0.7696
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input("Hours Studied", 0, 50, 20)
    attendance = st.number_input("Attendance (%)", 0, 100, 80)
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
    previous_scores = st.number_input("Previous Scores", 0, 100, 70)
    tutoring_sessions = st.number_input("Tutoring Sessions", 0, 20, 2)
    physical_activity = st.number_input("Physical Activity", 0, 10, 3)

with col2:
    motivation = st.selectbox(
        "Motivation Level",
        ["High", "Medium", "Low"],
    )
    family_income = st.selectbox(
        "Family Income",
        ["High", "Medium", "Low"],
    )
    internet = st.selectbox(
        "Internet Access",
        ["Yes", "No"],
    )
    teacher_quality = st.selectbox(
        "Teacher Quality",
        ["High", "Medium", "Low"],
    )
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
    )
    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"],
    )
    school_type = st.selectbox(
        "School Type",
        ["Private", "Public"],
    )

st.markdown("---")

parental_involvement = st.selectbox(
    "Parental Involvement",
    ["High", "Medium", "Low"],
)

access_resources = st.selectbox(
    "Access to Resources",
    ["High", "Medium", "Low"],
)

peer_influence = st.selectbox(
    "Peer Influence",
    ["Negative", "Neutral", "Positive"],
)

learning_disabilities = st.selectbox(
    "Learning Disabilities",
    ["No", "Yes"],
)

parent_education = st.selectbox(
    "Parental Education",
    ["College", "High School", "Postgraduate"],
)

distance = st.selectbox(
    "Distance From Home",
    ["Far", "Moderate", "Near"],
)

if st.button("Predict Exam Score"):
    data = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,

        "Parental_Involvement_Low": 1 if parental_involvement == "Low" else 0,
        "Parental_Involvement_Medium": 1 if parental_involvement == "Medium" else 0,

        "Access_to_Resources_Low": 1 if access_resources == "Low" else 0,
        "Access_to_Resources_Medium": 1 if access_resources == "Medium" else 0,

        "Extracurricular_Activities_Yes": 1 if extracurricular == "Yes" else 0,

        "Motivation_Level_Low": 1 if motivation == "Low" else 0,
        "Motivation_Level_Medium": 1 if motivation == "Medium" else 0,

        "Internet_Access_Yes": 1 if internet == "Yes" else 0,

        "Family_Income_Low": 1 if family_income == "Low" else 0,
        "Family_Income_Medium": 1 if family_income == "Medium" else 0,

        "Teacher_Quality_Low": 1 if teacher_quality == "Low" else 0,
        "Teacher_Quality_Medium": 1 if teacher_quality == "Medium" else 0,

        "School_Type_Public": 1 if school_type == "Public" else 0,

        "Peer_Influence_Neutral": 1 if peer_influence == "Neutral" else 0,
        "Peer_Influence_Positive": 1 if peer_influence == "Positive" else 0,

        "Learning_Disabilities_Yes": 1 if learning_disabilities == "Yes" else 0,

        "Parental_Education_Level_High School":
            1 if parent_education == "High School" else 0,

        "Parental_Education_Level_Postgraduate":
            1 if parent_education == "Postgraduate" else 0,

        "Distance_from_Home_Moderate":
            1 if distance == "Moderate" else 0,

        "Distance_from_Home_Near":
            1 if distance == "Near" else 0,

        "Gender_Male": 1 if gender == "Male" else 0,
    }

    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)
    score = float(prediction[0])

    # Grade
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 50:
        grade = "D"
    else:
        grade = "E"


    # Feature importance from linear model coefficients
    feature_names = list(data.keys())
    coef = None
    try:
        coef = np.ravel(model.coef_)
    except Exception:
        coef = None

    pos_factors = []
    neg_factors = []
    if coef is not None and len(coef) == len(feature_names):
        pairs = list(zip(feature_names, coef))
        pairs_sorted_desc = sorted(pairs, key=lambda x: x[1], reverse=True)
        pairs_sorted_asc = sorted(pairs, key=lambda x: x[1])
        pos_factors = [f"{name} ({coef:+.3f})" for name, coef in pairs_sorted_desc[:3]]
        neg_factors = [f"{name} ({coef:+.3f})" for name, coef in pairs_sorted_asc[:3]]

    # Charts (gauge + radar)
    gauge_value = min(max(int(score), 0), 100)
    gauge_bg = pd.DataFrame({"value": [100]})
    gauge_value_df = pd.DataFrame({"value": [gauge_value]})
    gauge_chart = alt.layer(
        alt.Chart(gauge_bg)
        .mark_arc(innerRadius=70, outerRadius=90, color="#e5e7eb")
        .encode(theta=alt.Theta("value:Q", scale=alt.Scale(domain=[0, 100]))),
        alt.Chart(gauge_value_df)
        .mark_arc(innerRadius=70, outerRadius=90, color="#2563eb")
        .encode(theta=alt.Theta("value:Q", scale=alt.Scale(domain=[0, 100])))
    ).properties(width=300, height=300)

    bar_df = pd.DataFrame(
        [
            {"Category": "Hours Studied", "Value": hours_studied / 50 * 100},
            {"Category": "Attendance", "Value": attendance},
            {"Category": "Sleep Hours", "Value": sleep_hours / 24 * 100},
            {"Category": "Previous Scores", "Value": previous_scores},
            {"Category": "Tutoring Sessions", "Value": tutoring_sessions / 20 * 100},
            {"Category": "Physical Activity", "Value": physical_activity / 10 * 100},
        ]
    )

    bar_chart = (
        alt.Chart(bar_df)
        .mark_bar(color="#4510f3")
        .encode(
            x=alt.X("Category:N", sort=None),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Value:Q")],
        )
        .properties(height=360, width=700)
    )

    # Session history (keep last 5)
    if "history" not in st.session_state:
        st.session_state.history = []
    entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({"time": entry_time, "score": f"{score:.2f}", "grade": grade})
    st.session_state.history = st.session_state.history[-5:]

    # Layout and display
    score_col, chart_col = st.columns([1, 1])

    with score_col:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
                        padding: 1.6rem;
                        border-radius: 24px;
                        color: white;
                        box-shadow: 0px 14px 35px rgba(15, 23, 42, 0.35);'>
                <h2 style='margin: 0 0 0.5rem 0;'>Predicted Exam Score</h2>
                <p style='font-size: 3rem; margin: 0; font-weight: 800;'>{score:.2f}</p>
                <p style='margin: 0.75rem 0 0 0; font-size: 1.2rem;'>Grade: <strong>{grade}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("\n")
        st.info("Use the chart to compare scores and study habits at a glance.")

    with chart_col:
        st.altair_chart(gauge_chart, use_container_width=True)

    st.progress(gauge_value)
    st.subheader("Performance Breakdown")
    st.altair_chart(bar_chart, use_container_width=True)

    # Feature importance display
    with st.expander("Feature Importance"):
        if pos_factors and neg_factors:
            st.write("Top Positive Factors:")
            for i, f in enumerate(pos_factors, 1):
                st.write(f"{i}. {f}")
            st.write("")
            st.write("Top Negative Factors:")
            for i, f in enumerate(neg_factors, 1):
                st.write(f"{i}. {f}")
        else:
            st.write("Feature importance not available for this model.")

    # Prediction history
    with st.expander("Prediction History (last 5)"):
        hist_df = pd.DataFrame(st.session_state.history)
        st.table(hist_df)

    # Download report
    report = (
        f"Predicted Score: {score:.2f}\n"
        f"Grade: {grade}\n"
        f"Time: {entry_time}\n"
    )
    st.download_button("Download Report", report, file_name="prediction_report.txt", mime="text/plain")

    if score >= 85:
        st.success("Excellent Performance")
    elif score >= 70:
        st.info("Good Performance")
    elif score >= 50:
        st.warning("Average Performance")
    else:
        st.error("Needs Improvement")

    st.subheader("Academic Insight")

    if attendance < 60:
        st.write(
            "Attendance is below recommended levels. Improving class participation can help boost scores."
        )
    elif hours_studied < 10:
        st.write(
            "Study time is low. Aim for more consistent study sessions to increase performance."
        )
    else:
        st.write(
            "Current habits look solid. Keep focusing on consistency and active learning."
        )

    if motivation == "Low":
        st.write(
            "Consider setting smaller study goals to maintain motivation and focus."
        )
    elif motivation == "High":
        st.write("Motivation level is strong — leverage it for steady improvement.")

    st.markdown("### Recommendations")
    recommendations = []

    if attendance < 60:
        recommendations.append("Improve attendance by joining class more regularly.")
    if hours_studied < 10:
        recommendations.append("Increase weekly study hours with a consistent routine.")
    if motivation == "Low":
        recommendations.append("Set short daily goals to keep study momentum high.")
    if family_income == "Low":
        recommendations.append("Explore free online resources to support learning.")
    if teacher_quality == "Low":
        recommendations.append("Seek extra tutoring or peer study groups for difficult topics.")

    if not recommendations:
        st.write("You're on the right track—keep maintaining consistent study habits.")
    else:
        for item in recommendations:
            st.write(f"- {item}")

    st.markdown("---")
    st.caption(
        "Student Performance Prediction System\n\nBuilt By Farel Yamotaro Hia\n• Streamlit\n• Scikit-Learn\n• Linear Regression"
    )
