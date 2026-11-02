"""
AutoGen Multi-Agent Customer Support Chatbot
Run: streamlit run app.py

Agents (5 total):
1. CategoryAnalysisAgent - classifies and routes each customer question.
2. AITechnologyAgent - AI/technical capabilities and integrations.
3. ProductReviewFeedbackAgent - product reviews, complaints, criticism and feedback.
4. ProductEnhancementAgent - feature requests and improvement suggestions.
5. PricingIssuesSupportAgent - pricing, billing, troubleshooting, issues and bugs.

Memory:
- SQLite stores normalized questions, assigned category and final answer.
- Exact matches are returned immediately.
- Similarity matching uses Python's standard-library difflib (no embedding service required).
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "customer_support_memory.db"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

AGENT_SPECS = {
    "AI Technology": {
        "name": "AITechnologyAgent",
        "description": "AI technology, technical capabilities, integrations, architecture and usage questions.",
        "system": (
            "You are the AI Technology customer support specialist. Answer questions about AI features, "
            "technical capabilities, integrations, setup, compatibility and usage. Be practical and clear. "
            "Do not invent product capabilities, integrations, versions or technical facts. If details are "
            "missing, state what must be confirmed and ask one focused question. Do not claim to have performed "
            "an action. End your response with the exact word End."
        ),
    },
    "Product Review & Feedback": {
        "name": "ProductReviewFeedbackAgent",
        "description": "Product reviews, customer criticism, complaints, satisfaction and feedback.",
        "system": (
            "You are the Product Review and Feedback specialist. Understand the customer's experience, "
            "acknowledge concerns respectfully, identify the main feedback, and offer a useful next step. "
            "Do not argue with the customer or fabricate review statistics. End your response with the exact word End."
        ),
    },
    "Product Enhancement": {
        "name": "ProductEnhancementAgent",
        "description": "Feature requests, usability improvements, roadmap suggestions and enhancement ideas.",
        "system": (
            "You are the Product Enhancement specialist. Clarify the requested improvement, explain possible "
            "value and practical considerations, and format feature requests clearly. Never promise that a "
            "feature will be delivered or invent roadmap commitments. End your response with the exact word End."
        ),
    },
    "Pricing, Issues & Bugs": {
        "name": "PricingIssuesSupportAgent",
        "description": "Pricing, plans, billing, payments, product issues, troubleshooting and bug reports.",
        "system": (
            "You are the Pricing, Issues and Bugs support specialist. Handle plan/pricing questions, billing "
            "concerns, troubleshooting and bug reports. Never invent prices, refund policies, account details "
            "or fixes. Ask for relevant non-sensitive details such as product area, error text and steps to "
            "reproduce. Never request passwords, API keys or full payment-card details. End your response "
            "with the exact word End."
        ),
    },
}

ROUTER_SYSTEM = """You are CategoryAnalysisAgent for a customer support system.
Classify the customer's message into exactly ONE category:
- AI Technology
- Product Review & Feedback
- Product Enhancement
- Pricing, Issues & Bugs

Routing rules:
- AI/ML, integrations, setup, technical capability, API or architecture -> AI Technology
- Complaint, review, criticism, satisfaction, general feedback -> Product Review & Feedback
- New feature, change request, usability improvement, roadmap idea -> Product Enhancement
- Price, plan, subscription, invoice, payment, error, defect, not working, troubleshooting -> Pricing, Issues & Bugs
If multiple categories apply, choose the category most directly related to the customer's main request.
Return only the category name, with no explanation or punctuation.
"""

# ---------- Logging ----------
class ActivityHandler(logging.Handler):
    """Send log records to both the terminal and Streamlit session activity feed."""
    def emit(self, record):
        try:
            message = self.format(record)
            print(message, flush=True)
            if "agent_activity" in st.session_state:
                st.session_state.agent_activity.append(message)
        except Exception:
            self.handleError(record)


def configure_logging():
    logger = logging.getLogger("customer_support_agents")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    handler = ActivityHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = configure_logging()

# ---------- Persistent memory ----------
def normalize_question(question: str) -> str:
    question = question.lower().strip()
    question = re.sub(r"\s+", " ", question)
    question = re.sub(r"[^\w\s]", "", question)
    return question


def question_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def init_memory():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS support_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT UNIQUE NOT NULL,
                normalized_question TEXT NOT NULL,
                original_question TEXT NOT NULL,
                category TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_memory_match(question: str, threshold: float = 0.94):
    normalized = normalize_question(question)
    if not normalized:
        return None

    with sqlite3.connect(DB_PATH) as conn:
        exact = conn.execute(
            "SELECT category, answer, original_question FROM support_memory WHERE question_hash = ?",
            (question_hash(normalized),),
        ).fetchone()
        if exact:
            return {"category": exact[0], "answer": exact[1], "matched_question": exact[2], "match_type": "exact"}

        rows = conn.execute(
            "SELECT normalized_question, category, answer, original_question "
            "FROM support_memory ORDER BY id DESC LIMIT 500"
        ).fetchall()

    best = None
    best_score = 0.0
    for saved_normalized, category, answer, original in rows:
        score = difflib.SequenceMatcher(None, normalized, saved_normalized).ratio()
        if score > best_score:
            best_score = score
            best = (category, answer, original)

    if best and best_score >= threshold:
        return {
            "category": best[0],
            "answer": best[1],
            "matched_question": best[2],
            "match_type": f"similar ({best_score:.0%})",
        }
    return None


def save_memory(question: str, category: str, answer: str):
    normalized = normalize_question(question)
    if not normalized:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO support_memory
            (question_hash, normalized_question, original_question, category, answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                question_hash(normalized),
                normalized,
                question.strip(),
                category,
                answer,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def get_memory_count() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT COUNT(*) FROM support_memory").fetchone()[0]


def clear_memory():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM support_memory")
        conn.commit()


# ---------- AutoGen / OpenAI ----------
def create_model_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your .env file or environment variables."
        )

    # OpenAIChatCompletionClient uses the official OpenAI API endpoint by default.
    # No custom base_url is needed when connecting directly to OpenAI.
    return OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.3,
    )


def extract_result_text(result) -> str:
    """Get the final text from an AutoGen TaskResult."""
    for message in reversed(result.messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return str(result)


def clean_category(raw: str) -> str:
    candidate = raw.strip().strip('"').strip("'").strip().rstrip(".")
    for category in AGENT_SPECS:
        if candidate.lower() == category.lower():
            return category

    # Fallback if the model adds a short explanation around the category.
    lower = candidate.lower()
    for category in AGENT_SPECS:
        if category.lower() in lower:
            return category
    LOGGER.warning("Router returned an unrecognized category: %r. Using support fallback.", raw)
    return "Pricing, Issues & Bugs"


async def run_customer_support(question: str):
    client = create_model_client()
    try:
        router = AssistantAgent(
            name="CategoryAnalysisAgent",
            model_client=client,
            system_message=ROUTER_SYSTEM,
            model_client_stream=False,
        )

        LOGGER.info("CategoryAnalysisAgent | ANALYZING customer question")
        route_result = await router.run(task=question)
        raw_category = extract_result_text(route_result)
        category = clean_category(raw_category)
        LOGGER.info("CategoryAnalysisAgent | ROUTED to %s", category)

        spec = AGENT_SPECS[category]
        specialist = AssistantAgent(
            name=spec["name"],
            model_client=client,
            system_message=spec["system"],
            model_client_stream=False,
        )

        LOGGER.info("%s | TASK ASSIGNED: %s", spec["name"], spec["description"])
        specialist_result = await specialist.run(
            task=(
                f"Customer question:\n{question}\n\n"
                f"Your assigned support category is: {category}.\n"
                "Respond directly to the customer. Keep the response helpful and concise."
            )
        )
        answer = extract_result_text(specialist_result).strip()
        answer = re.sub(r"\s*End\s*$", "", answer, flags=re.IGNORECASE).rstrip()
        answer = f"{answer}\n\nEnd"
        LOGGER.info("%s | RESPONSE COMPLETED", spec["name"])
        return category, answer
    finally:
        await client.close()


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="AutoGen Customer Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_memory()

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f6f8fc 0%, #eef3fb 100%); }
    .main .block-container { max-width: 1180px; padding-top: 1.8rem; }
    .hero {
        padding: 1.35rem 1.6rem; border-radius: 18px;
        background: linear-gradient(110deg, #14213d, #274c77);
        color: white; margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(20,33,61,.14);
    }
    .hero h1 { color: white; margin: 0; font-size: 1.8rem; }
    .hero p { color: #dce8f7; margin: .45rem 0 0 0; }
    div[data-testid="stChatMessage"] {
        border: 1px solid #e5eaf2; border-radius: 14px;
        padding: .55rem .8rem; background: rgba(255,255,255,.86);
    }
    .small-muted { color: #6b7280; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🤖 Support Control Center")
    st.caption("AutoGen multi-agent workflow powered by OpenAI")
    st.metric("Saved memory entries", get_memory_count())
    st.markdown("---")
    st.markdown("### Active agents")
    st.markdown(
        "- **CategoryAnalysisAgent** — routes questions\n"
        "- **AITechnologyAgent** — AI & technology\n"
        "- **ProductReviewFeedbackAgent** — reviews & feedback\n"
        "- **ProductEnhancementAgent** — feature requests\n"
        "- **PricingIssuesSupportAgent** — pricing, issues & bugs"
    )
    st.markdown("---")
    st.caption(f"Model: `{MODEL_NAME}`")
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_activity = []
        st.rerun()
    if st.button("Delete saved memory", use_container_width=True):
        clear_memory()
        st.success("Saved question memory cleared.")
        st.rerun()

st.markdown(
    """
    <div class="hero">
      <h1>Customer Support AI</h1>
      <p>Ask a question. The category agent analyzes it and assigns the right specialist.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_activity" not in st.session_state:
    st.session_state.agent_activity = []

chat_col, activity_col = st.columns([1.75, 1], gap="large")

with chat_col:
    st.subheader("💬 Support conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("category"):
                st.caption(f"Assigned agent: {message['category']}")
            st.markdown(message["content"])

    prompt = st.chat_input("Describe your question, feedback or product issue...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                cached = get_memory_match(prompt)
                if cached:
                    category = cached["category"]
                    answer = cached["answer"]
                    LOGGER.info(
                        "MEMORY | %s match found for previous question: %s",
                        cached["match_type"],
                        cached["matched_question"],
                    )
                    response_placeholder.markdown(
                        f"**Assigned agent:** {category}  \n"
                        f"*Retrieved from saved memory ({cached['match_type']} match)*\n\n{answer}"
                    )
                else:
                    with st.spinner("Category agent is analyzing and assigning a specialist..."):
                        category, answer = asyncio.run(run_customer_support(prompt))
                    save_memory(prompt, category, answer)
                    response_placeholder.markdown(f"**Assigned agent:** {category}\n\n{answer}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "category": category}
                )
            except Exception as exc:
                error_text = (
                    f"Sorry, I couldn't process this request. {exc}\n\n"
                    "Check your OpenAI API key, API billing, model access and package installation."
                )
                LOGGER.exception("WORKFLOW | Request failed")
                response_placeholder.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "category": "System"}
                )

with activity_col:
    st.subheader("🧭 Agent activity")
    st.caption("Live workflow logs are printed in the terminal and shown here.")
    if st.session_state.agent_activity:
        for line in reversed(st.session_state.agent_activity[-30:]):
            st.code(line, language="text")
    else:
        st.info("Agent activity will appear here after you send a message.")

st.markdown("---")
st.markdown(
    '<p class="small-muted">Prototype note: answers are generated by OpenAI. '
    'The SQLite memory is local to this project folder. Verify account-specific pricing, '
    'policies and technical commitments before sharing them with customers.</p>',
    unsafe_allow_html=True,
)
