import streamlit as st
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Intelligent Prompt Debugger",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Password gate
# --------------------------------------------------
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input(
            "Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False

    if not st.session_state["authenticated"]:
        st.text_input(
            "Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("Incorrect password")
        return False

    return True


if not check_password():
    st.stop()

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("Intelligent Prompt Debugger")
st.caption("Diagnose why prompts fail before trying to fix them.")

prompt = st.text_area(
    "Paste your prompt below",
    placeholder="e.g. Write a marketing email for my new product.",
    height=200
)

depth = st.selectbox(
    "Depth",
    ["Beginner", "Intermediate", "Advanced"]
)

# --------------------------------------------------
# Prompt failure zones
# --------------------------------------------------
ZONES = {
    "Intent Drift": [
        "What would a disappointing response look like?",
        "If this worked perfectly, what would you do next?",
        "What outcome are you actually hoping for?"
    ],
    "Missing Constraints": [
        "What format do you actually need?",
        "What must NOT appear in the response?",
        "How detailed is too detailed?"
    ],
    "Hidden Assumptions": [
        "What context exists only in your head?",
        "What definitions are you taking for granted?",
        "What knowledge are you assuming the model has?"
    ],
    "Audience Confusion": [
        "Who is this response for, specifically?",
        "What would make it unusable for them?",
        "What do they already know or believe?"
    ],
    "Evaluation Blindness": [
        "How will you know this response worked?",
        "What would make you discard it immediately?",
        "What criteria matter more than everything else?"
    ]
}

# Reduce questions for Beginner mode
if depth == "Beginner":
    ZONES = {k: v[:2] for k, v in ZONES.items()}

# --------------------------------------------------
# Semantic relevance (TF-IDF)
# --------------------------------------------------
@st.cache_resource
def build_vectorizer(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors


if st.button("Debug prompt"):
    if not prompt.strip():
        st.warning("Paste a prompt first.")
    else:
        zone_names = list(ZONES.keys())

        vectorizer, zone_vectors = build_vectorizer(zone_names)
        prompt_vector = vectorizer.transform([prompt])

        similarities = cosine_similarity(prompt_vector, zone_vectors)[0]
        ranked_indices = np.argsort(similarities)[::-1]

        st.subheader("Likely weak areas")

        for idx in ranked_indices[:3]:
            zone = zone_names[idx]
            st.markdown(f"### {zone}")
            for q in ZONES[zone]:
                st.write("•", q)

        st.divider()
        st.caption(
            "This tool does not rewrite prompts. It helps you clarify intent, constraints, and evaluation."
        )
