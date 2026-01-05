import streamlit as st
from datetime import date

# =================================================
# 🌟 Page Configuration
# =================================================
st.set_page_config(page_title="DevCard", page_icon="💻", layout="centered")

# =================================================
# 🎨 Custom Styling (CSS)
# =================================================
st.markdown(
    """
    <style>
        .title {
            font-size: 2.3rem;
            font-weight: bold;
            text-align: center;
            color: #4B8BBE;
        }
        .box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =================================================
# 🏷️ App Title & Intro
# =================================================
st.markdown("<div class='title'>👨‍💻 DevCard</div>", unsafe_allow_html=True)
st.caption("Minimal Streamlit DevCard 🚀")

# =================================================
# 🔁 Session State Initialization
# =================================================
if "start" not in st.session_state:
    st.session_state.start = False

# =================================================
# 🚀 Start Screen
# =================================================
if not st.session_state.start:
    if st.button("🚀 Start"):
        st.session_state.start = True
        st.balloons()
        st.rerun()

# =================================================
# 📋 Main Form Flow
# =================================================
if st.session_state.start:

    progress = st.progress(0)

    # ---------------------------------------------
    # ☑️ Agreement Section
    # ---------------------------------------------
    with st.container():
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        agree = st.checkbox("I agree to continue")
        st.markdown("</div>", unsafe_allow_html=True)

        if agree:
            progress.progress(30)

    # ---------------------------------------------
    # 🧑‍💻 Developer Preferences
    # ---------------------------------------------
    if agree:
        lang = st.radio("Language", ["Python", "JavaScript", "C++"])
        lib = st.selectbox("Library", ["NumPy", "Pandas", "Matplotlib"])
        level = st.slider("Skill Level", 0, 10, 5)
        progress.progress(60)

        # -----------------------------------------
        # 📝 Personal Details
        # -----------------------------------------
        name = st.text_input("Your Name")
        dob = st.date_input("Date of Birth", min_value=date(1900, 1, 1))
        num = st.number_input("Favorite Number", min_value=0, max_value=100, value=50)
        progress.progress(90)

        # -----------------------------------------
        # 🎉 Profile Preview & Download
        # -----------------------------------------
        if name:
            st.success("Profile Completed 🎉")

            col1, col2 = st.columns(2)
            col1.metric("Skill Level", level)
            col1.markdown(f"**Language:** {lang}")

            col2.markdown(f"**Name:** {name}")
            col2.markdown(f"**DOB:** {dob}")

            st.download_button(
                "📥 Download Profile",
                data=(
                    f"Name: {name}\n"
                    f"Language: {lang}\n"
                    f"Library: {lib}\n"
                    f"Level: {level}\n"
                    f"DOB: {dob}\n"
                    f"Favorite Number: {num}"
                ),
                file_name=f"{name}_profile.txt",
            )
            progress.progress(100)

    # ---------------------------------------------
    # 🔄 Reset App
    # ---------------------------------------------
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# =================================================
# 🌳 Project Structure (Tree View)
# =================================================
# DevCard_App/
# ├── app.py                 # Main Streamlit app
# ├── requirements.txt       # streamlit
# └── README.md              # App description

# =================================================
# 🔻 Footer
# =================================================
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")

# =================================================
# ✅ Summary
# =================================================
# • Custom CSS styling inside Streamlit
# • Session-state driven multi-step flow
# • Progress bar for user journey
# • Core widgets: checkbox, radio, selectbox, slider
# • Input widgets: text, date, number
# • Profile preview with metrics & columns
# • Downloadable DevCard profile
# • Reset functionality for fresh start
