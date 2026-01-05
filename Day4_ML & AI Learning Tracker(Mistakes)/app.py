import streamlit as st
from datetime import date
import pandas as pd

# =================================================
# 🌟 PAGE CONFIGURATION
# =================================================
st.set_page_config(
    page_title="ML & AI Learning Tracker",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML & AI Learning Tracker")
st.caption("Track your learning, avoid common mistakes, and stay on course!")
st.markdown("---")

# =================================================
# 🗂️ SESSION STATE INITIALIZATION
# =================================================
if "progress" not in st.session_state:
    st.session_state.progress = pd.DataFrame(
        columns=["Date", "Hours Spent", "Topics Covered", "Projects Completed"]
    )

if "mistakes_checked" not in st.session_state:
    st.session_state.mistakes_checked = []

# =================================================
# 📅 SELECT START DATE
# =================================================
st.header("📅 Select Your Learning Start Date")
start_date = st.date_input("Start Date", date.today())
st.markdown("---")

# =================================================
# ❌ COMMON MISTAKES SECTION
# =================================================
st.header("📝 Common Mistakes")
mistakes = [
    "Lack of Clear Goals",
    "Skipping Fundamentals",
    "Over-reliance on Libraries",
    "Insufficient Practice",
    "Ignoring Data Quality",
    "Neglecting Model Evaluation",
    "Avoiding Continuous Learning",
    "Poor Time Management",
    "Lack of Community Engagement",
    "Fear of Failure",
]

with st.expander("❌ Click to view common mistakes"):
    st.session_state.mistakes_checked = st.multiselect(
        "Select mistakes to track:", mistakes, default=[]
    )

# =================================================
# 💡 TIPS BASED ON MISTAKES
# =================================================
st.header("💡 How to Avoid These Mistakes")
tips = {
    "Lack of Clear Goals": "Set specific objectives for your ML & AI learning journey.",
    "Skipping Fundamentals": "Learn statistics, linear algebra, and Python basics before diving deep.",
    "Over-reliance on Libraries": "Understand the theory behind the algorithms.",
    "Insufficient Practice": "Build projects and apply concepts regularly.",
    "Ignoring Data Quality": "Focus on cleaning and preprocessing datasets properly.",
    "Neglecting Model Evaluation": "Always evaluate your models with proper metrics.",
    "Avoiding Continuous Learning": "Follow blogs, papers, and courses regularly.",
    "Poor Time Management": "Plan your weekly schedule and stick to it.",
    "Lack of Community Engagement": "Join forums, study groups, and networking events.",
    "Fear of Failure": "Experiment freely and learn from mistakes.",
}

for m in st.session_state.mistakes_checked:
    st.info(f"✅ {m}: {tips[m]}")

st.markdown("---")

# =================================================
# 📊 LEARNING TRACKER FORM
# =================================================
st.header("📊 Track Your Learning Progress")
with st.form("tracker_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        t_date = st.date_input("Date", date.today())
    with col2:
        hours = st.number_input("Hours Spent", min_value=0, max_value=24, value=1)
    with col3:
        topics = st.text_input("Topics Covered")
    with col4:
        projects = st.text_input("Projects Completed")

    submitted = st.form_submit_button("Add Entry")
    if submitted:
        new_row = pd.DataFrame(
            [[t_date, hours, topics, projects]],
            columns=["Date", "Hours Spent", "Topics Covered", "Projects Completed"],
        )
        st.session_state.progress = pd.concat(
            [st.session_state.progress, new_row], ignore_index=True
        )
        st.success("✅ Entry added successfully!")

# =================================================
# 📈 DISPLAY LEARNING PROGRESS
# =================================================
if not st.session_state.progress.empty:
    st.subheader("📈 Your Learning Progress")
    st.dataframe(st.session_state.progress)

    st.subheader("📊 Hours Spent Over Time")
    chart_data = (
        st.session_state.progress.groupby("Date")["Hours Spent"].sum().reset_index()
    )
    st.line_chart(chart_data.rename(columns={"Date": "index"}).set_index("index"))

    st.subheader("📊 Topics Covered Summary")
    topics_count = st.session_state.progress["Topics Covered"].value_counts()
    if not topics_count.empty:
        st.bar_chart(topics_count)

    # Download CSV
    st.download_button(
        "📥 Download Your Progress",
        data=st.session_state.progress.to_csv(index=False),
        file_name="ML_AI_Progress.csv",
    )

# =================================================
# 🔗 RESOURCES & FILE UPLOAD
# =================================================
st.subheader("🔗 Useful Dashboards and Tools")
tools_md = """
- [Kaggle](https://www.kaggle.com/) - Datasets and competitions.
- [Google Colab](https://colab.research.google.com/) - Free Jupyter notebooks in the cloud. 
- [TensorFlow Playground](https://playground.tensorflow.org/) - Visualize neural networks.
- [MLflow](https://mlflow.org) - Manage the ML lifecycle.
- [Weights & Biases](https://wandb.ai/) - Experiment tracking and model management
"""
st.markdown(tools_md)

file = st.file_uploader("Upload your own dataset (CSV only)", type=["csv"])
if file:
    dataset = pd.read_csv(file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(dataset)

    if not dataset.empty:
        st.subheader("Filter Dataset")
        column = st.selectbox("Select Column to Filter", dataset.columns)
        unique_values = dataset[column].unique()
        selected_value = st.selectbox("Select Value", unique_values)
        filtered_data = dataset[dataset[column] == selected_value]
        st.dataframe(filtered_data)
        st.success("✅ Dataset filtered successfully!")

# =================================================
# 🌳 PROJECT STRUCTURE (TREE VIEW)
# =================================================
# ML_AI_Tracker/
# ├── app.py                  # Main Streamlit app
# ├── requirements.txt        # pandas, streamlit
# └── README.md               # App description

# =================================================
# 🔻 FOOTER
# =================================================
st.markdown("---")
st.caption("🚀 Keep learning. Keep building. Developed with ❤️ using Streamlit.")

# =================================================
# ✅ SUMMARY
# =================================================
# • Initialize page layout with set_page_config
# • Session state tracks progress and mistakes
# • Track daily learning: date, hours, topics, projects
# • Tips displayed dynamically for selected mistakes
# • Progress charts: line chart & topics bar chart
# • CSV download & optional dataset upload with filtering
# • Resource links for ML & AI tools
