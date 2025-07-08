import streamlit as st
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# New 3D emoji image URL (dark, realistic)
emoji_url = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

# CSS Styling
st.markdown(f"""
    <style>
    body, .stApp {{
        background: linear-gradient(135deg, #e1f5fe, #b3e5fc);
        background-attachment: fixed;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url('https://www.transparenttextures.com/patterns/white-wall.png');
        opacity: 0.2;
        z-index: -1;
    }}

    .title {{
        text-align: center;
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px #ccc;
    }}

    .title img {{
        width: 42px;
        height: 42px;
        vertical-align: middle;
        margin-right: 12px;
        margin-bottom: 5px;
    }}

    .subtitle {{
        text-align: center;
        color: #8e24aa;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 1px #e0e0e0;
    }}

    .footer {{
        margin-top: 3rem;
        text-align: center;
        font-size: 0.9rem;
        color: #555;
    }}

    .custom-label {{
        font-size: 1.4rem;
        font-weight: 700;
        color: #0d47a1;
        margin-bottom: 0.5rem;
    }}

    .stTextInput > div > div > input {{
        background-color: #ffffff;
        color: #0d47a1;
        font-weight: 600;
        border-radius: 8px;
        border: 2px solid black !important;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }}

    .stButton > button {{
        background-color: #2196f3;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: 0.2s ease-in-out;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }}

    .stButton > button:hover {{
        background-color: #1976d2;
    }}

    .stAlert {{
        border: 2px solid black !important;
        border-radius: 10px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Title with new 3D emoji
st.markdown(f'<div class="title"><img src="{emoji_url}" alt="emoji">Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Understand the emotion behind your text using Machine Learning</div>', unsafe_allow_html=True)

# Custom label
st.markdown('<div class="custom-label">Enter your message or review:</div>', unsafe_allow_html=True)
review = st.text_input(label="", key="text_input", placeholder="E.g., This product made my day!")

# Prediction logic
if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before predicting.")
    else:
        review_scaled = scaler.transform([review]).toarray()
        result = model.predict(review_scaled)

        if result[0] == 0:
            st.error("🙁 It's a Negative Review", icon="❌")
        else:
            st.success("😊 It's a Positive Review", icon="✅")

# Footer
# Continuously scrolling footer with styled name
st.markdown("""
    <style>
    .scrolling-footer-wrapper {
        overflow: hidden;
        white-space: nowrap;
        width: 100%;
        margin-top: 3rem;
    }

    .scrolling-footer {
        display: inline-block;
        padding-left: 100%;
        animation: scroll-left 15s linear infinite;
        font-style: italic;
        font-weight: 900;
        font-size: 1.1rem;
        font-family: 'Poppins', 'Segoe UI', sans-serif;
        color: #2c3e50;
        text-shadow: 0.5px 0.5px 1px #ccc;
    }

    .footer-name {
        color: #800000; /* Dark maroon */
        font-weight: 900;
    }

    @keyframes scroll-left {
        0% {
            transform: translateX(0%);
        }
        100% {
            transform: translateX(-100%);
        }
    }
    </style>

    <div class="scrolling-footer-wrapper">
        <div class="scrolling-footer">
            Designed and Developed by <span class="footer-name">Saurabh Kumar Maurya</span> • Designed and Developed by <span class="footer-name">Saurabh Kumar Maurya</span> • 
        </div>
    </div>
""", unsafe_allow_html=True)
