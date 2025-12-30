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
# Prompt archetypes (training examples)
# --------------------------------------------------
PROMPT_PATTERNS = [
    {
        "type": "Instruction",
        "example": "Explain the following concept step by step for a beginner.",
        "template": (
            "Explain {topic} clearly and step by step.\n\n"
            "Audience: {audience}\n"
            "Tone: {tone}\n"
            "Include examples where helpful.\n"
            "Avoid unnecessary jargon."
        ),
    },
    {
        "type": "Creative",
        "example": "Write a short story about a character facing a moral dilemma.",
        "template": (
            "Write a {length} piece in the style of {style}.\n\n"
            "Subject: {topic}\n"
            "Mood: {tone}\n"
            "Constraints: {constraints}"
        ),
    },
    {
        "type": "Analytical",
        "example": "Compare two approaches and explain their pros and cons.",
        "template": (
            "Analyse {topic}.\n\n"
            "Compare: {items}\n"
            "Criteria: strengths, weaknesses, trade-offs\n"
            "Conclusion: practical recommendation"
        ),
    },
    {
        "type": "Vague",
        "example": "Tell me something about productivity.",
        "template": (
            "Provide a focused response on {topic}.\n\n"
            "Goal: {goal}\n"
            "Audience: {audience}\n"
            "Format: bullet points or steps\n"
            "Depth: practical and concrete"
        ),
    },
]


# --------------------------------------------------
# Build ML vectorizer
# --------------------------------------------------
@st.cache_resource
def build_vectorizer():
    texts = [p["example"] for p in PROMPT_PATTERNS]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors


vectorizer, pattern_vectors = build_vectorizer()


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Intelligent Prompt Debugger")
st.caption("Turn vague prompts into precise, high-quality instructions.")

user_prompt = st.text_area(
    "Paste your prompt",
    height=180,
    placeholder="e.g. explain productivity"
)

if st.button("Debug & Improve Prompt"):
    if not user_prompt.strip():
        st.warning("Paste a prompt first.")
    else:
        # Vectorise input
        input_vec = vectorizer.transform([user_prompt])
        similarities = cosine_similarity(input_vec, pattern_vectors)[0]

        best_idx = int(np.argmax(similarities))
        matched_pattern = PROMPT_PATTERNS[best_idx]

        # Output
        st.subheader("Improved Prompt")

        improved_prompt = matched_pattern["template"].format(
            topic="<<define topic clearly>>",
            audience="<<who this is for>>",
            tone="<<desired tone>>",
            length="<<length>>",
            style="<<style>>",
            constraints="<<any constraints>>",
            items="<<items to compare>>",
            goal="<<what you want to achieve>>"
        )

        st.text_area(
            "Refined prompt (copy & edit)",
            improved_prompt,
            height=220
        )

        st.divider()

        st.caption(
            f"Detected prompt type: **{matched_pattern['type']}**  \n"
            "This version adds structure, constraints, and clarity so the model knows exactly what to do."
        )
