import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "data"

# Load model dan scaler (robust for deployment)
linear_path = MODEL_DIR / "linear_regression.pkl"
scaler_path = MODEL_DIR / "scaler.pkl"

if not linear_path.exists() or not scaler_path.exists():
    st.error(
        "Model artifacts are missing.\n"
        f"Expected: {linear_path}\n"
        f"Expected: {scaler_path}"
    )
    st.info(f"Computed BASE_DIR: {BASE_DIR}")
    st.info(f"Computed MODEL_DIR: {MODEL_DIR}")
    st.stop()

model = joblib.load(linear_path)
scaler = joblib.load(scaler_path)


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

    background: linear-gradient(
        135deg,
        #1e293b,
        #334155
    );

    padding: 3rem;

    border-radius: 28px;

    text-align: center;

    box-shadow:
    0 15px 35px rgba(15,23,42,0.20);

    margin-bottom: 2rem;

    position: relative;

    overflow: hidden;
}

.hero-card::before{

    content:"";

    position:absolute;

    width:350px;
    height:350px;

    background:
    radial-gradient(
        rgba(255,255,255,0.08),
        transparent
    );

    top:-180px;
    right:-120px;
}

/* =========================
   HERO TEXT
========================= */

.hero-card *{
    color:#ffffff !important;
}

.hero-title{

    color:#ffffff !important;

    font-size:3rem;

    font-weight:800;

    margin-bottom:0.5rem;

    letter-spacing:-1px;
}

.hero-subtitle{

    color:rgba(255,255,255,0.9) !important;

    font-size:1rem;
}

.hero-stat-title{

    color:rgba(255,255,255,0.75) !important;

    font-size:0.9rem;

    font-weight:500;
}

.hero-stat-value{

    color:#ffffff !important;

    font-size:1.2rem;

    font-weight:700;
}

.hero-badge{
    display:inline-flex;
    align-items:center;
    gap:0.5rem;
    padding:0.65rem 1rem;
    border-radius:999px;
    background:rgba(56,189,248,0.16);
    color:#bfdbfe;
    font-weight:700;
    font-size:0.95rem;
    margin-top:1rem;
}

.hero-info-grid{
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    gap:1rem;
    margin-top:1.6rem;
}

.hero-stat-card{
    padding:1.2rem 1.3rem;
    border-radius:22px;
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(148,163,184,0.18);
    box-shadow:0 20px 40px rgba(255,255,255,0.05);
}

.hero-stat-card h3{
    margin:0;
    font-size:1.15rem;
    color:#ffffff;
    font-weight:800;
}

.hero-stat-card p{
    margin:0.55rem 0 0 0;
    color:rgba(226,232,240,0.82);
    line-height:1.55;
}

.hero-graphic{
    display:grid;
    grid-template-columns:repeat(3, minmax(0, 1fr));
    gap:1rem;
    margin-top:1.5rem;
}

.hero-graphic-card{
    padding:1rem;
    border-radius:20px;
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(148,163,184,0.12);
    color:#ffffff;
    min-height:130px;
}

.hero-graphic-card strong{
    display:block;
    font-size:1.2rem;
    margin-bottom:0.35rem;
}

.hero-graphic-card span{
    color:rgba(226,232,240,0.76);
    font-size:0.9rem;
}

/* =========================
   INPUTS
========================= */

[data-baseweb="input"]{

    border-radius:14px !important;

    border:1px solid rgba(148,163,184,0.2) !important;
}

.stSelectbox > div > div{

    border-radius:14px !important;
}

label{

    font-weight:600 !important;
}

/* =========================
   BUTTON
========================= */

.stButton > button{

    width:100%;

    height:58px;

    border:none;

    border-radius:16px;

    font-size:18px;

    font-weight:700;

    color:white;

    background:linear-gradient(
        135deg,
        #334155,
        #475569
    );

    transition:all .3s ease;

    box-shadow:
    0 10px 20px rgba(51,65,85,0.20);
}

.stButton > button:hover{

    transform:translateY(-2px);

    background:linear-gradient(
        135deg,
        #1e293b,
        #334155
    );

    box-shadow:
    0 15px 30px rgba(30,41,59,0.25);
}

/* =========================
   METRIC CARD
========================= */

[data-testid="metric-container"]{

    background:rgba(255,255,255,0.65);

    backdrop-filter:blur(15px);

    border-radius:18px;

    padding:1rem;

    border:1px solid rgba(148,163,184,0.15);

    transition:all .3s ease;
}

[data-testid="metric-container"]:hover{

    transform:translateY(-4px);

    box-shadow:
    0 10px 25px rgba(15,23,42,0.08);
}

/* =========================
   SIDEBAR
========================= */

[data-testid="stSidebar"]{

    background:
    linear-gradient(
        180deg,
        #0f172a,
        #1e293b
    );
}

[data-testid="stSidebar"] *{

    color:white;
}

/* =========================
   SIDEBAR CARD
========================= */

.sidebar-card{

    background:
    linear-gradient(
        135deg,
        #1e293b,
        #334155
    );

    padding:1.5rem;

    border-radius:18px;

    color:white;

    box-shadow:
    0 10px 25px rgba(15,23,42,0.20);
}

/* =========================
   EXPANDER
========================= */

.streamlit-expanderHeader{

    font-weight:700;
}

/* =========================
   TABLE
========================= */

[data-testid="stDataFrame"]{

    border-radius:18px;

    overflow:hidden;
}

/* =========================
   SCROLLBAR
========================= */

::-webkit-scrollbar{

    width:10px;
}

::-webkit-scrollbar-thumb{

    background:
    linear-gradient(
        #475569,
        #64748b
    );

    border-radius:999px;
}

/* =========================
   HORIZONTAL LINE
========================= */

hr{

    border:none;

    height:1px;

    background:rgba(148,163,184,0.2);
}

.sidebar-header{
    padding:1rem 1rem 0.75rem 1rem;
    margin-bottom:1rem;
    border-radius:20px;
    background:linear-gradient(180deg, rgba(56,189,248,0.18), rgba(59,130,246,0.08));
    border:1px solid rgba(255,255,255,0.12);
}

.sidebar-label{
    display:inline-flex;
    align-items:center;
    gap:0.5rem;
    padding:0.35rem 0.75rem;
    border-radius:999px;
    background:rgba(56,189,248,0.15);
    color:#bae6fd;
    font-weight:700;
    font-size:0.85rem;
}

.sidebar-title{
    margin:0;
    font-size:1.2rem;
    font-weight:800;
    color:#ffffff;
}

.sidebar-subtitle{
    margin:0.35rem 0 0 0;
    font-size:0.9rem;
    color:rgba(226,232,240,0.78);
}

.sidebar-card-inner{
    padding:1rem;
    border-radius:20px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:1rem;
}

.sidebar-metric-grid{
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    gap:0.8rem;
}

.sidebar-metric-box{
    padding:1rem;
    border-radius:18px;
    background:rgba(59,130,246,0.14);
    border:1px solid rgba(96,165,250,0.18);
}

.sidebar-metric-value{
    font-size:1.5rem;
    font-weight:800;
    color:#ffffff;
    margin-bottom:0.25rem;
}

.sidebar-metric-label{
    font-size:0.85rem;
    color:rgba(226,232,240,0.75);
}

.sidebar-section-title{
    font-size:0.95rem;
    font-weight:700;
    color:#e2e8f0;
    margin-bottom:0.5rem;
}

.sidebar-small-text{
    color:rgba(226,232,240,0.72);
    font-size:0.9rem;
    line-height:1.5;
}

.sidebar-footer-text{
    color:rgba(226,232,240,0.62);
    font-size:0.8rem;
}

</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sidebar-header">
    <div>
        <div class="sidebar-label">Live Analytics</div>
        <h1 class="sidebar-title">Student Insights</h1>
        <p class="sidebar-subtitle">Smart predictions • data-driven guidance</p>
    </div>
</div>

<div class="sidebar-card-inner">
    <div class="sidebar-metric-grid">
        <div class="sidebar-metric-box">
            <div class="sidebar-metric-value">76.96%</div>
            <div class="sidebar-metric-label">Model R²</div>
        </div>
        <div class="sidebar-metric-box">
            <div class="sidebar-metric-value">6,607</div>
            <div class="sidebar-metric-label">Dataset Size</div>
        </div>
    </div>
</div>

<div class="sidebar-card-inner">
    <p class="sidebar-section-title">Model Snapshot</p>
    <p class="sidebar-small-text">Linear Regression • 27 features • Predict exam performance from academic, social, and demographic data.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Model Info
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info("""
**Algorithm:** Linear Regression

**Features:** 27 Variables

**Target:** Exam Scores

**Training Data:** 6,607 Students
""")

# Sidebar - Quick Guide
with st.sidebar.expander("How to Use", expanded=False):
    st.markdown("""
1. **Enter Student Name** - Identify the student
2. **Fill Input Fields** - Provide academic and personal details
3. **Click Predict** - Generate exam score prediction
4. **View Results** - See score, grade, and insights
5. **Download Report** - Save prediction as text file
    """)

# Sidebar - Input Categories
with st.sidebar.expander("Input Categories", expanded=False):
    st.markdown("""
**Academic Factors:**
- Hours Studied
- Attendance
- Previous Scores
- Tutoring Sessions

**Health & Lifestyle:**
- Sleep Hours
- Physical Activity

**Social & Support:**
- Motivation Level
- Parental Involvement
- Peer Influence
- Extracurricular Activities

**Socioeconomic:**
- Family Income
- Parental Education
- Access to Resources

**Environmental:**
- Internet Access
- Teacher Quality
- School Type
- Distance From Home
    """)

# Sidebar - Tips
with st.sidebar.expander("Pro Tips", expanded=False):
    st.markdown("""
✓ **Accuracy Tip:** Fill all fields accurately for better predictions

✓ **Pattern:** Students with consistent study habits and good attendance typically score higher

✓ **Factors:** Sleep, motivation, and teacher quality significantly impact performance

✓ **History:** Track multiple predictions to see patterns
    """)

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

# Sidebar - Footer
st.sidebar.markdown("---")
st.sidebar.caption("""
**Student Performance Prediction System**

v1.0 | Built By Farel Yamotaro Hia PROJECT MACHINE LEARNING GDGoC USU 2026

🔧 Technologies: Streamlit • Scikit-Learn
""")

hero_html = """
<style>
.hero-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 2.4rem;
    border-radius: 28px;
    color: #ffffff;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.3);
    font-family: Inter, sans-serif;
    width: 100%;
    box-sizing: border-box;
}
.hero-title {
    margin: 0;
    font-size: 2.45rem;
    font-weight: 800;
    letter-spacing: -0.7px;
}
.hero-subtitle {
    margin: 0.75rem 0 1.25rem 0;
    color: rgba(226, 232, 240, 0.9);
    font-size: 1rem;
    max-width: 740px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.7rem 1rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.16);
    color: #bfdbfe;
    font-weight: 700;
    font-size: 0.96rem;
    margin-bottom: 1.6rem;
}
.hero-info-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.hero-stat-card {
    padding: 1.2rem 1.3rem;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 20px 40px rgba(255, 255, 255, 0.05);
}
.hero-stat-card h3 {
    margin: 0;
    font-size: 1.15rem;
    color: #ffffff;
    font-weight: 800;
}
.hero-stat-card p {
    margin: 0.55rem 0 0;
    color: rgba(226, 232, 240, 0.82);
    line-height: 1.6;
}
.hero-graphic {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1.25rem;
}
.hero-graphic-card {
    padding: 1rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(148, 163, 184, 0.12);
    color: #ffffff;
    min-height: 130px;
}
.hero-graphic-card strong {
    display: block;
    font-size: 1.1rem;
    margin-bottom: 0.4rem;
}
.hero-graphic-card span {
    color: rgba(226, 232, 240, 0.76);
    font-size: 0.9rem;
    line-height: 1.6;
}
@media (max-width: 1024px) {
    .hero-info-grid {
        grid-template-columns: 1fr;
    }
    .hero-graphic {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 720px) {
    .hero-card {
        padding: 1.4rem;
    }
    .hero-title {
        font-size: 2rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        margin-bottom: 1rem;
        max-width: 100%;
    }
    .hero-badge {
        font-size: 0.9rem;
        padding: 0.6rem 0.9rem;
        margin-bottom: 1rem;
    }
    .hero-graphic {
        grid-template-columns: 1fr;
    }
}
body { margin: 0; }
</style>
<div class="hero-card">
    <h1 class="hero-title">Student Performance Prediction</h1>
    <p class="hero-subtitle">Predicting exam scores based on academic, social, and demographic factors.</p>
    <div class="hero-badge">📈 Data-driven learning insights</div>

    <div class="hero-info-grid">
        <div class="hero-stat-card">
            <h3>76.96% Predictive Accuracy</h3>
            <p>Linear Regression was trained on 6,607 students for reliable score estimates.</p>
        </div>
        <div class="hero-stat-card">
            <h3>27 Insightful Inputs</h3>
            <p>Mix of academic, social, and lifestyle features to capture real student performance.</p>
        </div>
    </div>

    <div class="hero-graphic">
        <div class="hero-graphic-card">
            <strong>Study Focus</strong>
            <span>Visualizes the student’s preparation and support factors.</span>
        </div>
        <div class="hero-graphic-card">
            <strong>Performance Boost</strong>
            <span>Shows how attendance, motivation, and resources impact scores.</span>
        </div>
        <div class="hero-graphic-card">
            <strong>Smart Report</strong>
            <span>Generates a polished result with recommendation and downloadable report.</span>
        </div>
    </div>
</div>
"""

components.html(hero_html, height=700, scrolling=True)

st.markdown("---")

# Student Name Input
student_name = st.text_input("Student Name", placeholder="Enter student name")

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
    st.session_state.history.append({"name": student_name if student_name else "N/A", "time": entry_time, "score": f"{score:.2f}", "grade": grade})
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
                <p style='font-size: 1rem; margin: 0 0 1rem 0; opacity: 0.9;'><strong>Student:</strong> {student_name if student_name else "N/A"}</p>
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
        f"Student Name: {student_name if student_name else 'N/A'}\n"
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
        "Student Performance Prediction System\n\nBuilt By Farel Yamotaro Hia PROJECT MACHINE LEARNING GDGoC USU 2026\n• Streamlit\n• Scikit-Learn\n• Linear Regression"
    )
