from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# Page config
# ==================================================
st.set_page_config(
    page_title="Intelligent Prompt Debugger",
    page_icon="🧠",
    layout="centered"
)


# ==================================================
# Password gate
# ==================================================
def check_password() -> bool:
    def password_entered() -> None:
        expected = st.secrets.get("app_password", "")
        given = st.session_state.get("password", "")
        if expected and given == expected:
            st.session_state["authenticated"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False

    if not st.session_state["authenticated"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False

    return True


if not check_password():
    st.stop()


# ==================================================
# Storage (learning loop)
# ==================================================
STORE_PATH = "learning_store.json"

DEFAULT_STORE = {
    "playbook_stats": {},     # playbook -> {up:int, down:int}
    "examples": [],           # list of {"text": str, "label": str, "playbook": str}
    "feedback_log": []        # list of feedback events
}


def _safe_load_store() -> dict:
    if not os.path.exists(STORE_PATH):
        return json.loads(json.dumps(DEFAULT_STORE))
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ensure keys exist
        for k, v in DEFAULT_STORE.items():
            if k not in data:
                data[k] = json.loads(json.dumps(v))
        return data
    except Exception:
        # fallback to empty if corrupted
        return json.loads(json.dumps(DEFAULT_STORE))


def _safe_save_store(data: dict) -> None:
    try:
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        # On Streamlit Cloud, file writes usually work, but if they don't, fail gracefully
        pass


@st.cache_resource
def load_store_cached() -> dict:
    return _safe_load_store()


def get_store() -> dict:
    # Keep store in session for quick updates, but persist to file when changed.
    if "learning_store" not in st.session_state:
        st.session_state["learning_store"] = load_store_cached()
    return st.session_state["learning_store"]


def persist_store() -> None:
    data = st.session_state.get("learning_store")
    if isinstance(data, dict):
        _safe_save_store(data)


def bump_playbook_stat(playbook: str, up: bool) -> None:
    store = get_store()
    stats = store.setdefault("playbook_stats", {})
    pb = stats.setdefault(playbook, {"up": 0, "down": 0})
    if up:
        pb["up"] += 1
    else:
        pb["down"] += 1
    persist_store()


def log_feedback(event: dict) -> None:
    store = get_store()
    store.setdefault("feedback_log", []).append(event)
    # keep log from growing forever
    if len(store["feedback_log"]) > 500:
        store["feedback_log"] = store["feedback_log"][-500:]
    persist_store()


def add_example(text: str, label: str, playbook: str) -> None:
    store = get_store()
    ex = store.setdefault("examples", [])
    ex.append({"text": text.strip(), "label": label, "playbook": playbook})
    # cap
    if len(ex) > 300:
        store["examples"] = ex[-300:]
    persist_store()


# ==================================================
# Anonymiser (best-effort redaction of obvious identifiers)
# ==================================================
RE_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
RE_PHONE = re.compile(r"\b(\+?\d[\d\s().-]{7,}\d)\b")
RE_HANDLE = re.compile(r"(?<!\w)@[\w_]{2,}\b")
RE_BRACKET_TAGS = re.compile(r"\[[^\]]{1,80}\]")  # short bracket tags like [BOOT SEQUENCE...]

def anonymise_text(text: str) -> str:
    t = text
    t = RE_EMAIL.sub("[REDACTED_EMAIL]", t)
    t = RE_URL.sub("[REDACTED_URL]", t)
    t = RE_HANDLE.sub("[REDACTED_HANDLE]", t)
    t = RE_PHONE.sub("[REDACTED_PHONE]", t)
    # Remove unique bracket tags (optional but often identifying)
    t = RE_BRACKET_TAGS.sub("[REDACTED_TAG]", t)
    return t


# ==================================================
# ML archetypes + scaffolds
# ==================================================
@dataclass(frozen=True)
class Archetype:
    name: str
    playbook: str
    examples: List[str]


SCAFFOLDS: Dict[str, str] = {
    "analysis": """You are an expert {role}.

Objective:
- {objective}

Context:
- {context}

Task:
- Analyse: {topic}
- Provide a clear, logically structured response.

Constraints:
- Tone: {tone}
- Audience: {audience}
- Length: {length}
- Must include: {must_include}
- Must avoid: {must_avoid}

Output format:
- 2–3 sentence summary first
- Headings + bullet points
- Trade-offs + edge cases
- End with a practical recommendation

Quality bar:
- Define terms precisely
- Make reasoning explicit
- No generic filler
""",
    "compare": """You are an expert {role}.

Objective:
- {objective}

Context:
- {context}

Task:
- Compare {item_a} vs {item_b} for: {topic}
- Evaluate using criteria: {criteria}

Constraints:
- Tone: {tone}
- Audience: {audience}
- Length: {length}
- Must include: {must_include}
- Must avoid: {must_avoid}

Output format:
- Table: criteria | A | B | winner | notes
- Then: “Choose A when…”, “Choose B when…”
- End with a recommendation + rationale

Quality bar:
- Be specific, not vague
- State assumptions
- Mention failure modes
""",
    "writing": """You are a {role}.

Objective:
- {objective}

Context:
- {context}

Task:
- Write: {topic}

Constraints:
- Tone: {tone}
- Audience: {audience}
- Length: {length}
- Must include: {must_include}
- Must avoid: {must_avoid}

Output format:
- Final copy only (no meta commentary)
- Strong structure, high readability
- Use concrete language

Quality bar:
- Every sentence earns its place
- No fluff
""",
    "coding": """You are a senior {role}.

Objective:
- {objective}

Context:
- {context}

Task:
- Build: {topic}

Constraints:
- Stack: {stack}
- Compatibility: {compat}
- Complexity: {complexity}
- Must include: {must_include}
- Must avoid: {must_avoid}

Output format:
- Code in one block
- Minimal necessary comments
- Include usage instructions + small example
- Error handling + edge cases

Quality bar:
- Clean architecture
- Sensible defaults
- No unnecessary dependencies
""",
    "prompting": """You are a prompt engineer and technical editor.

Objective:
- {objective}

Context:
- {context}

Task:
- Convert the user's intent into a high-performance prompt.

Constraints:
- Target behaviour: {target_behaviour}
- Tone: {tone}
- Audience: {audience}
- Length: {length}
- Must include: {must_include}
- Must avoid: {must_avoid}

Output format (follow exactly):
1) ROLE
2) INPUTS (required + optional)
3) INSTRUCTIONS (step-by-step)
4) OUTPUT SPEC (exact formatting)
5) QUALITY CHECKS (self-check list)

Quality bar:
- Specific, testable instructions
- No contradictions
- Clear stop conditions
"""
}


ARCHETYPES: List[Archetype] = [
    Archetype(
        name="Prompt engineering",
        playbook="prompting",
        examples=[
            "Improve this prompt so the AI produces better output.",
            "Rewrite my prompt to be precise, structured, and unambiguous.",
            "Turn this messy idea into a world-class prompt.",
            "Debug my prompt and rewrite it with a clear output format."
        ],
    ),
    Archetype(
        name="Analytical explanation",
        playbook="analysis",
        examples=[
            "Explain this step by step for a beginner.",
            "Give a structured analysis with recommendations.",
            "Break this down logically and propose next steps.",
            "Explain causes, trade-offs, and what to do."
        ],
    ),
    Archetype(
        name="Comparison / decision",
        playbook="compare",
        examples=[
            "Compare these two options and recommend one.",
            "Pros and cons of A vs B for my use case.",
            "Which is better and why? Use clear criteria.",
            "Evaluate alternatives using a decision table."
        ],
    ),
    Archetype(
        name="Writing / copy",
        playbook="writing",
        examples=[
            "Write a landing page section with a strong hook.",
            "Rewrite this to be clearer and more professional.",
            "Write a warm message that persuades without pressure.",
            "Write an email in a professional tone."
        ],
    ),
    Archetype(
        name="Coding / technical build",
        playbook="coding",
        examples=[
            "Build a Streamlit app with these features.",
            "Write Python code with error handling.",
            "Debug this code and propose a clean fix.",
            "Create a small tool and explain how to run it."
        ],
    ),
]


DEFAULTS = {
    "role": "prompt engineer + technical editor",
    "objective": "Produce a best-in-class result with minimal ambiguity.",
    "context": "The user wants a refined prompt that yields high-quality, repeatable output.",
    "tone": "professional and clear",
    "audience": "a competent creator or developer",
    "length": "concise but complete",
    "must_include": "clear structure, constraints, examples, edge cases",
    "must_avoid": "vagueness, filler, contradictions, ungrounded claims",
    "criteria": "cost, speed, quality, maintainability, risk",
    "stack": "Python + Streamlit",
    "compat": "Python 3.13 on Streamlit Community Cloud",
    "complexity": "production-ready but not over-engineered",
    "target_behaviour": "Ask only essential clarifying questions; otherwise make best-effort assumptions and label them."
}


# ==================================================
# Build classifier (base archetypes + learned examples)
# ==================================================
def build_corpus(store: dict) -> Tuple[List[str], List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    playbooks: List[str] = []

    for a in ARCHETYPES:
        for ex in a.examples:
            texts.append(ex)
            labels.append(a.name)
            playbooks.append(a.playbook)

    # Learned examples (from user submissions)
    for ex in store.get("examples", []):
        t = ex.get("text", "").strip()
        if t:
            texts.append(t)
            labels.append(ex.get("label", "Prompt engineering"))
            playbooks.append(ex.get("playbook", "prompting"))

    return texts, labels, playbooks


@st.cache_resource
def fit_vectorizer(initial_texts: List[str]):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(initial_texts)
    return vec, X


def refit_if_needed() -> None:
    # If learning examples changed, rebuild model for this session.
    store = get_store()
    fingerprint = len(store.get("examples", []))
    if st.session_state.get("corpus_fingerprint") != fingerprint:
        texts, labels, playbooks = build_corpus(store)
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        st.session_state["clf_vec"] = vec
        st.session_state["clf_X"] = X
        st.session_state["clf_labels"] = labels
        st.session_state["clf_playbooks"] = playbooks
        st.session_state["corpus_fingerprint"] = fingerprint


def classify_intent(prompt: str) -> Tuple[str, str, float]:
    refit_if_needed()
    vec = st.session_state.get("clf_vec")
    X = st.session_state.get("clf_X")
    labels = st.session_state.get("clf_labels")
    playbooks = st.session_state.get("clf_playbooks")

    if vec is None or X is None or labels is None or playbooks is None:
        store = get_store()
        texts, labels, playbooks = build_corpus(store)
        vec, X = fit_vectorizer(texts)
        st.session_state["clf_vec"] = vec
        st.session_state["clf_X"] = X
        st.session_state["clf_labels"] = labels
        st.session_state["clf_playbooks"] = playbooks

    if not prompt.strip():
        return "Prompt engineering", "prompting", 0.0

    v = vec.transform([prompt])
    sims = cosine_similarity(v, X)[0]
    idx = int(np.argmax(sims))
    return labels[idx], playbooks[idx], float(sims[idx])


# ==================================================
# Learning-aware playbook selection
# We bias toward playbooks that historically got upvoted.
# (Still constrained by ML prediction, but nudged.)
# ==================================================
def choose_playbook(pred_playbook: str) -> str:
    store = get_store()
    stats = store.get("playbook_stats", {})
    pb = stats.get(pred_playbook, {"up": 0, "down": 0})
    up = pb.get("up", 0)
    down = pb.get("down", 0)

    # If the predicted playbook has poor history, fall back to prompting
    # (conservative choice).
    if (down - up) >= 5 and pred_playbook != "prompting":
        return "prompting"
    return pred_playbook


# ==================================================
# Prompt builder
# ==================================================
def build_prompt(playbook: str, fields: Dict[str, str], original: str, include_original: bool) -> str:
    tmpl = SCAFFOLDS.get(playbook, SCAFFOLDS["prompting"])
    out = tmpl.format(
        role=fields["role"],
        objective=fields["objective"],
        context=fields["context"],
        topic=fields.get("topic", "the user’s request (define precisely)"),
        tone=fields["tone"],
        audience=fields["audience"],
        length=fields["length"],
        must_include=fields["must_include"],
        must_avoid=fields["must_avoid"],
        item_a=fields.get("item_a", "<<Option A>>"),
        item_b=fields.get("item_b", "<<Option B>>"),
        criteria=fields.get("criteria", fields["criteria"]),
        stack=fields["stack"],
        compat=fields["compat"],
        complexity=fields["complexity"],
        target_behaviour=fields["target_behaviour"],
    ).strip()

    if include_original:
        out += "\n\n---\nOriginal prompt (verbatim):\n" + original.strip()

    return out


# ==================================================
# UI
# ==================================================
st.title("Intelligent Prompt Debugger")
st.caption("Paste a prompt. Get back a structured, world-class prompt. Then rate it to train the system.")

SANITISED_SAMPLE = """[PROTOCOL RE-INITIATED]
[DOCTRINE FRAMEWORK v3 – TOTAL SPECTRUM INTEGRATION]
A modular creative operating system for structured thinking, writing, and analysis.

CORE DIRECTIVES:
1) Be precise: define terms and scope.
2) Be structured: headings, steps, checklists where useful.
3) Be honest about uncertainty: label assumptions clearly.
4) Prefer actionable output: practical next steps.
5) Avoid contradictions: resolve conflicts explicitly.

OUTPUT FORMATS:
- SITREP, OPLAN, DEBRIEF, RED TEAM, PSYPROF
"""

with st.expander("Tools: Anonymise text (best-effort)"):
    anon_in = st.text_area("Paste text to anonymise", value="", height=120)
    if st.button("Anonymise"):
        st.text_area("Anonymised output", value=anonymise_text(anon_in), height=120)

st.divider()

left, right = st.columns([2, 1])

with left:
    user_prompt = st.text_area(
        "Your prompt",
        height=220,
        value=st.session_state.get("last_prompt", SANITISED_SAMPLE),
        placeholder="Paste your prompt here..."
    )

with right:
    include_original = st.checkbox("Append original prompt", value=True)
    detail = st.selectbox("Detail level", ["Tight", "Standard", "Deep"], index=1)
    st.markdown("**Learning**")
    persist_note = "File-based learning is best-effort on Streamlit Cloud (may reset on redeploy)."
    st.caption(persist_note)

st.subheader("Optional tightening (big quality boost)")

c1, c2 = st.columns(2)
with c1:
    role = st.text_input("Role", value=DEFAULTS["role"])
    tone = st.selectbox("Tone", ["professional and clear", "direct", "warm", "formal"], index=0)
    audience = st.text_input("Audience", value=DEFAULTS["audience"])
with c2:
    objective = st.text_input("Objective", value=DEFAULTS["objective"])
    length = st.selectbox("Length", ["concise but complete", "short", "medium", "detailed"], index=0)
    must_include = st.text_input("Must include", value=DEFAULTS["must_include"])
    must_avoid = st.text_input("Must avoid", value=DEFAULTS["must_avoid"])

topic = st.text_input("Topic / task focus (one line)", value="")

with st.expander("If comparison prompt (optional)"):
    item_a = st.text_input("Option A", value="")
    item_b = st.text_input("Option B", value="")
    criteria = st.text_input("Criteria", value=DEFAULTS["criteria"])

with st.expander("Coding settings (optional)"):
    stack = st.text_input("Stack", value=DEFAULTS["stack"])
    compat = st.text_input("Compatibility", value=DEFAULTS["compat"])
    complexity = st.selectbox(
        "Complexity",
        ["production-ready but not over-engineered", "minimal", "enterprise"],
        index=0
    )

with st.expander("Advanced behaviour (optional)"):
    target_behaviour = st.text_area("Target behaviour", value=DEFAULTS["target_behaviour"], height=90)


def tune_depth(text: str) -> str:
    t = text.strip()
    if detail == "Tight":
        return t
    if detail == "Standard":
        return t + ", explicit output format, clear assumptions"
    return t + ", explicit output format, clear assumptions, validations/tests, edge-case handling"


if st.button("Generate world-class prompt"):
    if not user_prompt.strip():
        st.warning("Paste a prompt first.")
        st.stop()

    st.session_state["last_prompt"] = user_prompt

    label, pred_playbook, conf = classify_intent(user_prompt)
    playbook = choose_playbook(pred_playbook)

    fields = {
        "role": role.strip() or DEFAULTS["role"],
        "objective": objective.strip() or DEFAULTS["objective"],
        "context": DEFAULTS["context"],
        "tone": tone.strip() or DEFAULTS["tone"],
        "audience": audience.strip() or DEFAULTS["audience"],
        "length": length.strip() or DEFAULTS["length"],
        "must_include": tune_depth(must_include.strip() or DEFAULTS["must_include"]),
        "must_avoid": must_avoid.strip() or DEFAULTS["must_avoid"],
        "criteria": criteria.strip() or DEFAULTS["criteria"],
        "stack": stack.strip() or DEFAULTS["stack"],
        "compat": compat.strip() or DEFAULTS["compat"],
        "complexity": complexity.strip() or DEFAULTS["complexity"],
        "target_behaviour": target_behaviour.strip() or DEFAULTS["target_behaviour"],
    }

    if topic.strip():
        fields["topic"] = topic.strip()
    if item_a.strip():
        fields["item_a"] = item_a.strip()
    if item_b.strip():
        fields["item_b"] = item_b.strip()
    if criteria.strip():
        fields["criteria"] = criteria.strip()

    output = build_prompt(playbook, fields, user_prompt, include_original)
    st.session_state["last_output"] = output
    st.session_state["last_label"] = label
    st.session_state["last_playbook"] = playbook
    st.session_state["last_conf"] = conf

    st.subheader("Output: world-class prompt")
    st.text_area("Copy/paste prompt", value=output, height=320)

    with st.expander("Diagnostics (ML + learning)"):
        st.write(f"Detected intent: **{label}**")
        st.write(f"Predicted playbook: **{pred_playbook}**")
        st.write(f"Chosen playbook (learning-aware): **{playbook}**")
        st.write(f"Confidence: **{conf:.2f}**")
        store = get_store()
        stats = store.get("playbook_stats", {}).get(playbook, {"up": 0, "down": 0})
        st.write(f"Playbook history: 👍 {stats.get('up',0)} | 👎 {stats.get('down',0)}")


# ==================================================
# Learning loop UI (rate + submit improved prompt)
# ==================================================
if "last_output" in st.session_state:
    st.divider()
    st.subheader("Teach it (learning loop)")

    colA, colB, colC = st.columns([1, 1, 3])

    with colA:
        if st.button("👍 Good output"):
            bump_playbook_stat(st.session_state.get("last_playbook", "prompting"), up=True)
            log_feedback({
                "ts": datetime.utcnow().isoformat() + "Z",
                "type": "vote",
                "vote": "up",
                "playbook": st.session_state.get("last_playbook", "prompting"),
                "label": st.session_state.get("last_label", ""),
                "conf": st.session_state.get("last_conf", 0.0)
            })
            st.success("Saved. This playbook will be prioritised more often.")

    with colB:
        if st.button("👎 Needs work"):
            bump_playbook_stat(st.session_state.get("last_playbook", "prompting"), up=False)
            log_feedback({
                "ts": datetime.utcnow().isoformat() + "Z",
                "type": "vote",
                "vote": "down",
                "playbook": st.session_state.get("last_playbook", "prompting"),
                "label": st.session_state.get("last_label", ""),
                "conf": st.session_state.get("last_conf", 0.0)
            })
            st.warning("Saved. The app will bias away from this playbook when results are poor.")

    with colC:
        st.caption("Optional: paste an improved version of your prompt. This becomes a new training example (lightweight ML).")

    improved = st.text_area("Improved prompt (optional)", value="", height=130)

    lab = st.selectbox(
        "Label this example as",
        [a.name for a in ARCHETYPES],
        index=0
    )

    pb_map = {a.name: a.playbook for a in ARCHETYPES}
    pb_for_label = pb_map.get(lab, "prompting")

    if st.button("Add improved prompt to learning set"):
        if not improved.strip():
            st.error("Paste an improved prompt first.")
        else:
            add_example(improved, lab, pb_for_label)
            log_feedback({
                "ts": datetime.utcnow().isoformat() + "Z",
                "type": "example_added",
                "label": lab,
                "playbook": pb_for_label
            })
            st.success("Added. The classifier will adapt in this session, and best-effort persist to storage.")


# ==================================================
# Admin / stats
# ==================================================
with st.expander("Stats (learning store)"):
    store = get_store()
    st.write("Playbook stats:")
    st.code(json.dumps(store.get("playbook_stats", {}), indent=2))
    st.write(f"Learned examples: {len(store.get('examples', []))}")
    st.write(f"Feedback log events: {len(store.get('feedback_log', []))}")
