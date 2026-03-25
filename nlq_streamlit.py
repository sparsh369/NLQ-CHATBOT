import os
import logging
import sys
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langgraph.prebuilt import create_react_agent


# ---------------- CONFIG ----------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("app_log.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "inventory.db")
EXCEL_PATH = os.path.join(BASE_DIR,"Current Inventory.xlsx")  # ✅ FIXED

# Streamlit page config
st.set_page_config(
    page_title="Inventory NLQ Chatbot",
    page_icon="📦",
    layout="wide",
)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = None


# ---------------- LOAD DATA ----------------

def load_excel_to_sqlite():
    """Load Excel into SQLite if DB doesn't exist."""
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        logger.info("SQLite DB already exists, skipping load.")
        return

    if not os.path.exists(EXCEL_PATH):
        st.error(f"❌ Excel file not found at: {EXCEL_PATH}")
        st.stop()

    logger.info(f"Loading Excel from {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("inventory", engine, if_exists="replace", index=False)
    engine.dispose()

    logger.info(f"Data written to {DB_PATH}")


# ---------------- AGENT ----------------

@st.cache_resource
def initialize_agent():
    load_excel_to_sqlite()

    # DB
    engine = create_engine(f"sqlite:///{DB_PATH}")
    db = SQLDatabase(engine=engine)

    # ✅ STREAMLIT SECRETS (IMPORTANT)
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("⚠️ Please add OPENAI_API_KEY in Streamlit secrets.")
        st.stop()

    api_key = st.secrets["OPENAI_API_KEY"]

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_prompt = """You are a helpful data analyst that answers questions about an inventory database using SQL.

Rules:
1. First list tables and get the schema before writing queries.
2. Write correct SQLite queries.
3. Limit results to 20 rows unless user asks more.
4. NEVER run INSERT, UPDATE, DELETE, DROP.
5. Always explain results clearly.
6. If no results, say it clearly.
7. Always show the SQL query used.
"""

    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent


# ---------------- UI ----------------

def main():
    st.title("📦 Inventory NLQ Chatbot")
    st.markdown("Ask questions about your inventory data in plain English.")

    # Sidebar
    with st.sidebar:
        st.header("📊 Quick Questions")

        example_questions = [
            "What materials have the highest shelf stock?",
            "Show me all raw materials in plant 2001",
            "Which product families have the most demand?",
            "What is the total WIP value by material type?",
            "List materials where safety stock exceeds demand",
            "How many unique plants are there?",
        ]

        for q in example_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.prefill_query = q
                st.rerun()

        st.markdown("---")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.prefill_query = None
            st.rerun()

    # Initialize agent
    agent = initialize_agent()

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask a question about your inventory...")

    if st.session_state.prefill_query:
        user_input = st.session_state.prefill_query
        st.session_state.prefill_query = None

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]}
                    )

                    response = result["messages"][-1].content

                    st.markdown(response)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    logger.error(error_msg)

                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()
