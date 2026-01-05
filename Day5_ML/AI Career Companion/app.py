import streamlit as st
import pandas as pd
import altair as alt
from datetime import date, datetime
import plotly.graph_objects as go

# =================================================
# 🌟 PAGE CONFIGURATION
# =================================================
st.set_page_config(
    page_title="ML/AI Career Companion",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================================================
# 📊 SESSION STATE INIT
# =================================================
if "salary_history" not in st.session_state:
    st.session_state.salary_history = []
if "learning_log" not in st.session_state:
    st.session_state.learning_log = []
if "target_currency" not in st.session_state:
    st.session_state.target_currency = "EUR"

# =================================================
# 🎨 CUSTOM CSS STYLING
# =================================================
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 10px;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin: 10px 0;
}
.highlight {
    background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
    padding: 2px 6px;
    border-radius: 4px;
}
.skill-bar {
    height: 8px;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    border-radius: 4px;
    margin: 5px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# =================================================
# 🚀 SIDEBAR SETTINGS
# =================================================
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # 👤 User Profile
    st.subheader("Profile")
    user_name = st.text_input("Your Name", value="ML Enthusiast")
    experience_level = st.select_slider(
        "Experience Level", options=["Beginner", "Intermediate", "Advanced", "Expert"]
    )

    # 🎨 Theme Toggle
    st.markdown("---")
    st.subheader("Theme")
    dark_mode = st.toggle("Dark Mode", value=False)

    # 💾 Data Management
    st.markdown("---")
    st.subheader("Data")
    if st.button("Clear History"):
        st.session_state.salary_history.clear()
        st.session_state.learning_log.clear()
        st.success("All history cleared!")

    st.caption("Version 2.0 • Updated: Today")

# =================================================
# 🏆 MAIN HEADER
# =================================================
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    st.markdown(
        '<h1 class="main-header">🚀 ML/AI Career Companion</h1>', unsafe_allow_html=True
    )
    st.caption(f"Welcome, {user_name} | Level: {experience_level}")
with col3:
    st.metric("Today", datetime.now().strftime("%b %d"))

st.markdown("---")

# =================================================
# 💰 SALARY CONVERTER
# =================================================
st.header("💰 Salary Converter")
CURRENCY_DATA = {
    "EUR": {"rate": 0.92, "symbol": "€", "name": "Euro"},
    "GBP": {"rate": 0.81, "symbol": "£", "name": "British Pound"},
    "INR": {"rate": 83.5, "symbol": "₹", "name": "Indian Rupee"},
    "JPY": {"rate": 136.2, "symbol": "¥", "name": "Japanese Yen"},
    "AUD": {"rate": 1.57, "symbol": "A$", "name": "Australian Dollar"},
    "CAD": {"rate": 1.35, "symbol": "C$", "name": "Canadian Dollar"},
    "USD": {"rate": 1.0, "symbol": "$", "name": "US Dollar"},
}

# Salary input & conversion
col1, col2, col3 = st.columns(3)
with col1:
    salary_usd = st.number_input(
        "💵 Annual Salary (USD)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=100_000.0,
        step=5_000.0,
    )
with col2:
    currency_options = [k for k in CURRENCY_DATA if k != "USD"]
    target_currency = st.selectbox(
        "🎯 Convert to",
        currency_options,
        format_func=lambda x: f"{CURRENCY_DATA[x]['symbol']} {CURRENCY_DATA[x]['name']}",
    )
with col3:
    st.markdown("### ")
    if st.button("💾 Save Conversion", use_container_width=True):
        conversion = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "amount_usd": salary_usd,
            "currency": target_currency,
            "converted": salary_usd * CURRENCY_DATA[target_currency]["rate"],
        }
        st.session_state.salary_history.append(conversion)
        st.success("Conversion saved!")

converted_amount = salary_usd * CURRENCY_DATA[target_currency]["rate"]
symbol = CURRENCY_DATA[target_currency]["symbol"]

st.markdown(
    f"""
<div class="metric-card">
    <h2 style="margin:0; color:white; font-size: 2.5rem;">
        ${salary_usd:,.0f} USD = {symbol}{converted_amount:,.0f} {target_currency}
    </h2>
    <p style="margin:0; opacity:0.9;">
        Exchange Rate: 1 USD = {CURRENCY_DATA[target_currency]['rate']} {target_currency}
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# =================================================
# 📊 MARKET INSIGHTS
# =================================================
st.header("📈 Market Insights")
tab1, tab2, tab3 = st.tabs(["📊 Trends", "📈 Growth", "🎯 Opportunities"])

# ----- Trends -----
with tab1:
    job_data = pd.DataFrame(
        {
            "Year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "ML/AI Jobs": [
                80_000,
                110_000,
                145_000,
                190_000,
                240_000,
                300_000,
                370_000,
            ],
            "Avg Salary (USD)": [
                95_000,
                105_000,
                115_000,
                125_000,
                135_000,
                145_000,
                155_000,
            ],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=job_data["Year"],
            y=job_data["ML/AI Jobs"],
            mode="lines+markers",
            name="ML/AI Jobs",
            line=dict(color="#667eea", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=job_data["Year"],
            y=job_data["Avg Salary (USD)"],
            mode="lines+markers",
            name="Avg Salary (USD)",
            yaxis="y2",
            line=dict(color="#84fab0", width=3),
        )
    )
    fig.update_layout(
        title="ML/AI Job Market Growth",
        yaxis=dict(title="Number of Jobs"),
        yaxis2=dict(title="Avg Salary (USD)", overlaying="y", side="right"),
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

# ----- Growth -----
with tab2:
    col1, col2, col3 = st.columns(3)
    col1.metric("Annual Growth", "35%", "↗️ 5% from last year")
    col2.metric("Salary Premium", "28%", "vs. traditional tech roles")
    col3.metric("Remote Jobs", "45%", "of all ML/AI positions")

    skills_data = pd.DataFrame(
        {
            "Skill": [
                "Python",
                "TensorFlow",
                "PyTorch",
                "MLOps",
                "NLP",
                "Computer Vision",
                "RL",
            ],
            "Demand": [95, 85, 80, 75, 70, 65, 60],
            "Salary Impact": [15, 25, 28, 30, 35, 40, 45],
        }
    )
    skill_chart = (
        alt.Chart(skills_data)
        .mark_bar()
        .encode(
            x=alt.X("Demand:Q", title="Demand (%)"),
            y=alt.Y("Skill:N", sort="-x"),
            color=alt.Color("Salary Impact:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["Skill", "Demand", "Salary Impact"],
        )
        .properties(height=300, title="Top In-Demand Skills")
    )
    st.altair_chart(skill_chart, use_container_width=True)

# ----- Opportunities -----
with tab3:
    st.markdown(
        """
    - 🌟 Build portfolio projects to demonstrate skills
    - 🚀 Contribute to open-source ML/AI projects
    - 📚 Upskill with online courses and certifications
    - 🤝 Network with ML/AI professionals
    """
    )

# =================================================
# 🎓 LEARNING JOURNEY
# =================================================
st.header("📚 Learning Journey")
path = st.selectbox(
    "Choose your learning path:",
    [
        "🤖 Machine Learning Fundamentals",
        "🧠 Deep Learning Specialist",
        "🗣️ NLP Engineer",
        "👁️ Computer Vision",
        "⚡ MLOps Engineer",
    ],
)

roadmaps = {
    "🤖 Machine Learning Fundamentals": {
        "months": 3,
        "skills": ["Python", "Pandas", "Scikit-learn", "Statistics", "Linear Algebra"],
        "projects": [
            "House Price Prediction",
            "Customer Segmentation",
            "Spam Classifier",
        ],
    },
    "🧠 Deep Learning Specialist": {
        "months": 6,
        "skills": [
            "PyTorch/TensorFlow",
            "Neural Networks",
            "CNN",
            "RNN",
            "Transformers",
        ],
        "projects": ["Image Classifier", "Text Generator", "Style Transfer"],
    },
}

if path in roadmaps:
    roadmap = roadmaps[path]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Timeline")
        st.progress(min(roadmap["months"] / 12, 1.0))
        st.caption(f"Estimated: {roadmap['months']} months")
        st.subheader("🛠️ Key Skills")
        for skill in roadmap["skills"]:
            st.markdown(f"✅ {skill}")
    with col2:
        st.subheader("🚀 Project Ideas")
        for project in roadmap["projects"]:
            st.markdown(f"🔨 {project}")
        st.subheader("🎯 Daily Goal")
        goal = st.slider("Hours to study today", 0, 8, 2)
        if st.button("✅ Log Today's Study"):
            st.session_state.learning_log.append(
                {
                    "date": date.today().isoformat(),
                    "path": path,
                    "hours": goal,
                }
            )
            st.success(f"Logged {goal} hours of study!")

# =================================================
# 🎯 FOCUS & PROGRESS TRACKER
# =================================================
st.header("🎯 Focus & Progress")
focus_col1, focus_col2, focus_col3 = st.columns(3)
with focus_col1:
    focus_option = st.radio(
        "Current Focus Level:",
        [
            "🚀 Intense (90%+)",
            "📚 Steady (60-89%)",
            "⚖️ Balanced (30-59%)",
            "🌱 Exploring (<30%)",
        ],
    )
with focus_col2:
    age_group = st.select_slider(
        "Age Group", options=["18-22", "23-26", "27-30", "31-35", "36+"]
    )
with focus_col3:
    weekly_hours = st.slider("Weekly Study Hours", 0, 40, 10)

progress_data = pd.DataFrame(
    {
        "Category": [
            "Technical Skills",
            "Projects",
            "Theoretical Knowledge",
            "Industry Exposure",
        ],
        "Progress": [65, 40, 75, 30],
    }
)
st.altair_chart(
    alt.Chart(progress_data)
    .mark_bar()
    .encode(
        x="Progress:Q",
        y=alt.Y("Category:N", sort="-x"),
        color=alt.Color("Progress:Q", scale=alt.Scale(scheme="goldred")),
    )
    .properties(height=250, title="Learning Progress Breakdown"),
    use_container_width=True,
)

# =================================================
# 📝 LEARNING LOG
# =================================================
if st.session_state.learning_log:
    st.subheader("📖 Study History")
    st.dataframe(pd.DataFrame(st.session_state.learning_log), use_container_width=True)

# =================================================
# 💡 RESOURCES
# =================================================
with st.expander("🔗 Essential Resources"):
    resources = {
        "📚 Books": [
            "Hands-On Machine Learning - Aurélien Géron",
            "Deep Learning - Ian Goodfellow",
            "Pattern Recognition - Christopher Bishop",
        ],
        "🎓 Courses": [
            "Coursera: Machine Learning (Andrew Ng)",
            "Fast.ai: Practical Deep Learning",
            "Stanford CS229: Machine Learning",
        ],
        "🛠️ Tools": ["Jupyter Notebooks", "VS Code", "Git & GitHub", "Docker for ML"],
        "👥 Communities": [
            "Kaggle",
            "r/MachineLearning",
            "Towards Data Science",
            "MLOps Community",
        ],
    }
    for category, items in resources.items():
        st.markdown(f"**{category}**")
        for item in items:
            st.markdown(f"• {item}")
        st.markdown("")

# =================================================
# 🎯 ACTION PLAN
# =================================================
st.markdown("---")
st.subheader("🎯 Your Action Plan This Week")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        "1. Complete one small project\n"
        "2. Review linear algebra basics\n"
        "3. Join one ML community\n"
        "4. Study for 1 hour daily\n"
        "5. Read one technical article"
    )
with col2:
    st.markdown(
        "• Build portfolio with 3+ projects\n"
        "• Master one ML framework\n"
        "• Contribute to open-source\n"
        "• Network with 5+ professionals\n"
        "• Land internship/job in AI field"
    )

# =================================================
# 📊 QUICK STATS
# =================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Total Study Hours",
    sum(log.get("hours", 0) for log in st.session_state.learning_log),
)
col2.metric("Conversion History", len(st.session_state.salary_history))
col3.metric("Days Streak", "7")
col4.metric("Next Milestone", "30h")

# =================================================
# 🎉 DAILY MOTIVATION
# =================================================
st.markdown("---")
quote = st.selectbox(
    "💡 Today's Motivation",
    [
        "The only way to learn is to code and build!",
        "Every expert was once a beginner.",
        "Consistency beats intensity in the long run.",
        "Your neural network needs training too!",
        "The data doesn't lie, but it needs interpretation.",
    ],
)
st.success(f"**{quote}**")

# =================================================
# 🔻 FOOTER
# =================================================
st.markdown("---")
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.caption(
        "🚀 Keep pushing forward. Every line of code brings you closer to mastery."
    )
    st.caption("💡 Focus on understanding concepts, not just memorizing syntax.")
with footer_col2:
    st.caption("Made with ❤️ using Streamlit")
