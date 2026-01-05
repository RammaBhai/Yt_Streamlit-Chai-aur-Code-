import streamlit as st
from datetime import date
import pandas as pd

# =========================
# 🌟 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ML & AI Learning Journey",
    page_icon="🤖",
    layout="wide",
)

# =========================
# 🧠 TITLE
# =========================
st.title("🤖 My ML & AI Learning Dashboard")
st.caption("Machine Learning • Deep Learning • NLP • Computer Vision • RL")
st.markdown("---")

# =========================
# SESSION STATE INIT
# =========================
if "selections" not in st.session_state:
    st.session_state.selections = {}

# =========================
# LEARNING PACE INPUT
# =========================
st.sidebar.header("📊 Learning Plan")
months = st.sidebar.slider("Select your learning pace (months):", 1, 12, 6)
weekly_hours = st.sidebar.number_input("Weekly study hours:", 1, 40, 10)
project_hours = st.sidebar.number_input("Weekly project hours:", 0, 20, 5)
st.sidebar.markdown(
    f"**Pace:** {months} months | **Weekly Study:** {weekly_hours}h | **Projects:** {project_hours}h"
)

with st.expander("ℹ️ Learning Pace Info"):
    st.markdown(
        """
- 1-3 months: Intensive  
- 4-6 months: Moderate  
- 7-12 months: Relaxed
"""
    )

# =========================
# CONFIG FOR ML
# =========================
ml_topics = {
    "Supervised Learning": {
        "concepts": ["Classification", "Regression"],
        "algorithms": ["Decision Trees", "SVM", "Logistic Regression"],
    },
    "Unsupervised Learning": {
        "concepts": ["Clustering", "Dimensionality Reduction"],
        "algorithms": ["K-Means", "Hierarchical Clustering", "DBSCAN"],
    },
    "Reinforcement Learning": {
        "concepts": ["MDPs", "Q-Learning"],
        "algorithms": ["Q-Learning", "DQN", "Policy Gradients"],
    },
}

# =========================
# LAYOUT: Columns for Topics
# =========================
cols = st.columns(3)
for col, (topic, data) in zip(cols, ml_topics.items()):
    with col:
        st.subheader(f"🔍 {topic}")
        for concept in data["concepts"]:
            st.markdown(f"- {concept}")
        selection = st.radio(f"Select an algorithm:", data["algorithms"], key=topic)
        st.session_state.selections[topic] = selection
        st.caption(f"Selected: **{selection}**")

# =========================
# FEEDBACK
# =========================
st.markdown("---")
st.subheader("🧠 Learning Path Feedback")

sel = st.session_state.selections
if (
    sel.get("Supervised Learning") == "Decision Trees"
    and sel.get("Unsupervised Learning") == "K-Means"
    and sel.get("Reinforcement Learning") == "Q-Learning"
):
    st.success("🎯 Excellent foundational choices for ML mastery!")
    st.balloons()
elif (
    sel.get("Supervised Learning") == "SVM"
    and sel.get("Unsupervised Learning") == "Hierarchical Clustering"
    and sel.get("Reinforcement Learning") == "DQN"
):
    st.info("🚀 Strong advanced selections. Keep going!")
    st.snow()
else:
    st.warning(
        "⚠️ Consider mixing foundational and advanced algorithms for broader understanding."
    )

# =========================
# DEEP LEARNING, NLP, CV
# =========================
dl_topics = {
    "Deep Learning": ["Neural Networks", "CNNs", "RNNs", "Transformers"],
    "NLP": [
        "Tokenization",
        "Embeddings",
        "Language Models",
        "Sentiment Analysis",
        "NER",
    ],
    "Computer Vision": [
        "Image Classification",
        "Object Detection",
        "Semantic Segmentation",
        "Transfer Learning",
    ],
}

for topic, concepts in dl_topics.items():
    with st.expander(f"📘 {topic} Key Concepts"):
        for c in concepts:
            st.markdown(f"- {c}")

# =========================
# TOOLS & LIBRARIES
# =========================
tools = [
    "Python",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Seaborn",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Keras",
    "OpenCV",
]
with st.expander("🛠️ Tools & Libraries"):
    st.markdown("\n".join(f"- {t}" for t in tools))

# =========================
# ADDITIONAL TOPICS
# =========================
additional = [
    "Feature Engineering",
    "Model Evaluation",
    "Hyperparameter Tuning",
    "Ensemble Methods",
    "Time Series Analysis",
    "NLP & CV Applications",
    "RL Applications",
]
with st.expander("📚 Additional Topics"):
    st.markdown("\n".join(f"- {a}" for a in additional))

# =========================
# SUMMARY & EXPORT
# =========================
st.markdown("---")
st.subheader("📋 Learning Summary")

summary = {
    "Learning Pace (months)": months,
    "Weekly Study Hours": weekly_hours,
    "Project Hours per Week": project_hours,
    **sel,
}

df_summary = pd.DataFrame(summary.items(), columns=["Item", "Selection"])
st.table(df_summary)

st.download_button(
    "📥 Download Your Learning Plan",
    data="\n".join(f"{k}: {v}" for k, v in summary.items()),
    file_name="ML_AI_Learning_Plan.txt",
)

st.markdown("---")
st.caption("🚀 Keep learning. Keep building. Developed with ❤️ using Streamlit.")
