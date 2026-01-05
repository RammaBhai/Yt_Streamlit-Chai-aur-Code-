import streamlit as st

# ---------------------------
# 🌟 App Configuration
# ---------------------------
st.set_page_config(page_title="Chai aur Code", page_icon="☕💻", layout="centered")

# ---------------------------
# 🏷️ App Title & Introduction
# ---------------------------
st.title("Hello from Chai aur Code ☕💻")
st.subheader("Let's get started with Streamlit")
st.write(
    "Streamlit makes it **super easy** to create interactive web apps for "
    "data science and machine learning. 🚀"
)

# ---------------------------
# ☕ Chai Preferences Section
# ---------------------------
st.markdown("### ☕ Chai Time!")
like_chai = st.checkbox("Do you like Chai?")

if like_chai:
    chai_type = st.selectbox(
        "Choose your favorite type of Chai:",
        ["Masala Chai", "Ginger Chai", "Cardamom Chai", "Plain Chai"],
    )
    st.success(f"You selected: {chai_type} 🍵")

# ---------------------------
# 👤 User Name Input Section
# ---------------------------
st.markdown("### 👤 Tell us your name")
name = st.text_input("Enter your name:")

if name:
    st.info(f"Welcome, {name}! 🎉")

# ---------------------------
# 🎯 Footer / Fun Message
# ---------------------------
st.markdown("---")
st.write("Made with ❤️ by Chai aur Code")

# ---------------------------
# 🌳 Code Structure Overview (Tree Style)
# ---------------------------
# Chai_aur_Code_App/
# ├── streamlit_app.py       # Main app file
# ├── requirements.txt       # Dependencies (streamlit, etc.)
# └── README.md              # Project description

# ✅ Summary
# - Use st.set_page_config() for page title, icon, and layout
# - Use st.title(), st.subheader(), st.write() for text content
# - Use st.checkbox() for boolean input
# - Use st.selectbox() for multiple choice selection
# - Use st.text_input() for user input
# - Use st.success() and st.info() for feedback messages
# - Use st.markdown("---") for section dividers
