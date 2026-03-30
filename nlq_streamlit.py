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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("app_log.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "inventory.db")
EXCEL_PATH = os.path.join(BASE_DIR,"Current Inventory.xlsx")

st.set_page_config(
    page_title="Inventory NLQ Chatbot",
    page_icon="📦",
    layout="wide",
)

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

    # Strip trailing spaces from column names (e.g. 'Product Family ' -> 'Product Family')
    df.columns = [col.strip() for col in df.columns]

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("inventory", engine, if_exists="replace", index=False)
    engine.dispose()
    logger.info(f"Data written to {DB_PATH} — {len(df):,} rows, {len(df.columns)} columns")


# ---------------- SYSTEM PROMPT ----------------

def build_system_prompt() -> str:
    return """You are a helpful inventory data analyst. You answer questions by writing and running SQL
against a SQLite database. Think carefully before writing SQL — follow every rule below.

════════════════════════════════════════════════════════
DATABASE:  SQLite   TABLE: inventory   ROWS: 126,472
════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLUMN REFERENCE  (wrap EVERY column name in double-quotes in SQL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column Name            | Type    | What it means
-----------------------|---------|-----------------------------
"Plant"                | INTEGER | Plant/site ID e.g. 2001, 2024
"Material"             | TEXT    | Material code e.g. 363097-000
"Material Name"        | TEXT    | Full name of the material
"Material Type"        | TEXT    | Category of material (see values below)
"UOM"                  | TEXT    | Unit of measure e.g. FT, EA, KG
"Shelf Stock"          | REAL    | Quantity sitting on shelf
"Shelf Stock ($)"      | REAL    | Dollar value of shelf stock
"GIT"                  | REAL    | Goods in transit quantity
"GIT ($)"              | REAL    | Dollar value of GIT
"WIP"                  | REAL    | Work in progress quantity
"WIP($)"               | REAL    | Dollar value of WIP
"DOH"                  | REAL    | Days on hand
"Safety Stock"         | REAL    | Minimum stock to keep
"Demand"               | REAL    | Total demand quantity
"Product Family"       | TEXT    | Product family code e.g. ETL, HWAT
"SOP Family"           | TEXT    | SOP planning family e.g. SENSORS, FIBER
"Product Group"        | TEXT    | Detailed product group name
"Material Group"       | TEXT    | Material group e.g. Custom Cable
"Product Category"     | TEXT    | Category e.g. PD / Project
"Material Application" | TEXT    | Application e.g. KA / Floor Heating
"Sub Application"      | TEXT    | Sub-application e.g. KSA / Leak Detection
"ABC"                  | TEXT    | ABC classification: A, B, or C
"MRP Controller Text"  | TEXT    | MRP controller name/code
"Purchasing Group Text"| TEXT    | Purchasing group name

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWN COLUMN VALUES  (use LIKE for matching — never invent values)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Material Type" — 11 exact values:
  Raw materials | Semifinished products | Finished products | Trading goods
  Packaging | Operating supplies-NON VA | Nonvaluated materials
  Prod. resources/tools | Optng suppl/Non Cos-VALUA | Spare parts | Services

"Plant" — 48 numeric IDs (sample):
  2001, 2006, 2007, 2012, 2013, 2014, 2015, 2018, 2019, 2020,
  2021, 2022, 2023, 2024, 2025, 2026, 3001, 3002, 3003 ...

"ABC":  A | B | C

"SOP Family" — sample values:
  MONO | MONO-CEL_D | SENSORS | FIBER-ZONE | RWC-BO | FIBER-COAT
  SENSORS ROPED CABLES | SEN-BULK | Reynosa Sensors | CMPT
  SENSORS SUB ASSY | Reynosa FrostGuards | NUHEAT | SEN-KITT
  nVent Thermal Europe | SENSORS SUB EPOXY | FIBER | PKG

"Product Family" — sample values:
  ETL | XL-TRACE | HWAT | BTV | T2RED | ICESTOP | XTV | WGRD-H
  XPI | CMPTS-IHTS | QTVR | WGRD-FS | EM | TT SENSORS | VPL
  PLAB-SR | CCH | TRACETEK ACC/INSTR | JBS/JBM/T-100

"Product Category" — 13 exact values:
  PD / Project | PD / Polymer Pipe Heat Tracing - BIS
  PD / Heat Tracing Components | PD / Polymer Pipe Heat Tracing - IND
  PD / Floor Heating | PD / Snow Melting & De-Icing
  PD / Control, Monitoring & Power Distribution
  PD / Fire and Performance Wiring | PD / Leak Detection
  PD / MI Heat Tracing | PD / Discountinued Products
  PD / Tip Clearance/Gadolina | PD / Mscellaneous

"Material Application" — 11 exact values:
  KA / Commercial Heat-Tracing | KA / Industrial Heat-Tracing
  KA / Floor Heating | KA / Speciality Heating
  KA / Fire and Performance Wiring | KA / Leak Detection
  KA / OFS | KA / Temperature Measurement | KA / Tip Clearance
  KA / Rail and Transit Heating | KA / Gadolina

"Sub Application" — 19 exact values:
  KSA / Pipe Freeze Protection | KSA / Hot Water Temperature Maintenance
  KSA / Industrial Heat-Tracing | KSA / Floor Heating
  KSA / Roof & Gutter De-Icing | KSA / Commercial Components
  KSA / Speciality Heating | KSA / Surface Snow Melting
  KSA / Fire and Performance Wiring - BIS | KSA / Leak Detection
  KSA / Fire and Performance Wiring - IND | KSA / Downhole/Bottomhole Heating
  KSA / Project | KSA / In-Pipe Heating Cables
  KSA / Temperature Measurement | KSA / Tip Clearance
  KSA / Rail and Transit Heating | KSA / Oil Tank Freeze Protection
  KSA / Gadolina

"Material Group" — 271 unique values (sample):
  Custom Cable | Resins - Engineering Plastics - General | Cables - General
  Injection Molded - Plastic | Stamped Metal - Stamping | Rayclic
  Electro-Mechanical - General | Electrical Supplies
  Electronic Components - Connectors | Conductor ...

"MRP Controller Text" — 141 unique values (sample):
  SENSOR RAW-FIBER | FIBER SUBASSEMBLY | Mfg: I Plant, Elec
  Buy: I Plant, Elec | Buy: Panel Compon | Imported Material
  MPS Parts CN | SENSOR RAW-COMP | SENSOR WIP | INDUSTRIAL FG BO ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NULL / MISSING DATA RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: Several columns are mostly NULL in this dataset:
  "Product Family", "SOP Family", "Product Group",
  "Material Application", "Sub Application"

When filtering on these columns ALWAYS add:
  AND "column_name" IS NOT NULL
When aggregating these columns ALWAYS add:
  WHERE "column_name" IS NOT NULL
This prevents NULL rows from polluting your results.

Example:
  SELECT "SOP Family", SUM("Shelf Stock") AS total
  FROM inventory
  WHERE "SOP Family" IS NOT NULL
  GROUP BY "SOP Family"
  ORDER BY total DESC
  LIMIT 20;

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AMBIGUOUS KEYWORD RULE  (CRITICAL — read carefully)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Some words like "sensors" or "fiber" appear in MULTIPLE columns.
Before writing a WHERE clause for a vague keyword, ALWAYS search
across ALL relevant columns using OR, like this:

Example — user says "show me sensors":
  SELECT * FROM inventory
  WHERE (
    "SOP Family"          LIKE '%sensor%' OR
    "MRP Controller Text" LIKE '%sensor%' OR
    "Product Family"      LIKE '%sensor%' OR
    "Material Name"       LIKE '%sensor%' OR
    "Material Group"      LIKE '%sensor%'
  )
  LIMIT 20;

Then look at the results and explain which columns matched and why.
NEVER search only one column when the keyword could appear in many.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEX CALCULATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For ratio / derived metrics, use SQLite-safe expressions:

DOH vs Demand ratio by Plant:
  SELECT "Plant",
         ROUND(SUM("DOH"), 2)    AS total_DOH,
         ROUND(SUM("Demand"), 2) AS total_Demand,
         ROUND(
           CASE WHEN SUM("Demand") = 0 THEN NULL
                ELSE SUM("DOH") / SUM("Demand")
           END, 4
         ) AS doh_demand_ratio
  FROM inventory
  WHERE "Demand" IS NOT NULL AND "DOH" IS NOT NULL
  GROUP BY "Plant"
  ORDER BY doh_demand_ratio DESC
  LIMIT 20;

Safety Stock coverage (Safety Stock / Demand):
  ROUND(
    CASE WHEN "Demand" = 0 OR "Demand" IS NULL THEN NULL
         ELSE "Safety Stock" / "Demand"
    END, 4
  ) AS coverage_ratio

Rules for calculations:
- ALWAYS wrap divisions in a CASE WHEN denominator = 0 THEN NULL END
  to prevent divide-by-zero crashes.
- ALWAYS filter out NULLs with IS NOT NULL on columns used in math.
- ALWAYS use ROUND(..., 2) on all dollar and ratio results.
- For percentages multiply by 100 and label clearly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIFIC MATERIAL LOOKUP RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the user asks about a specific material code or name:
1. First try exact match on "Material":
   WHERE "Material" = 'FIBER-XVR-32'
2. If zero rows → try LIKE on "Material":
   WHERE "Material" LIKE '%FIBER-XVR%'
3. If still zero rows → try LIKE on "Material Name":
   WHERE "Material Name" LIKE '%FIBER-XVR-32%'
4. If still zero → tell the user clearly that this material
   does not exist in the database and suggest checking the spelling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHELF STOCK QUANTITY vs VALUE RULE  (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Shelf Stock" stores RAW QUANTITY (FT, ST, LB, EA, etc.).
Each row has a different UOM, so SUM("Shelf Stock") across
multiple materials is MEANINGLESS — you'd be adding feet to
pieces to pounds.

"Shelf Stock ($)" stores the DOLLAR VALUE and IS safe to SUM
across materials because it's always in the same unit ($).

RULES:
1. For ANY aggregation across multiple materials or plants,
   ALWAYS use SUM("Shelf Stock ($)") — never SUM("Shelf Stock").

2. Only use "Shelf Stock" (quantity) when:
   - Filtering a single material with a known UOM
   - The user explicitly asks for quantity/units (not value)
   - You are also showing the UOM column alongside it

3. When user asks "shelf stock available", "total shelf stock",
   "how much shelf stock" → default to "Shelf Stock ($)" and
   label the result clearly as dollar value.

4. When a user asks for both quantity AND value, return both
   columns separately — never combine them.

CORRECT example for "total sensor shelf stock":
  SELECT SUM("Shelf Stock ($)") AS total_shelf_stock_value
  FROM inventory
  WHERE "MRP Controller Text" LIKE '%SENSOR%'

WRONG example (never do this):
  SELECT SUM("Shelf Stock") AS total_shelf_stock   ← mixes units!
  FROM inventory
  WHERE "MRP Controller Text" LIKE '%SENSOR%'
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL SQL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ALWAYS wrap column names in double-quotes.
   ✅  SELECT "Shelf Stock ($)" FROM inventory
   ❌  SELECT Shelf Stock ($) FROM inventory

2. ALWAYS use LIKE '%value%' for text — never bare = for strings.
   ✅  WHERE "Material Type" LIKE '%Raw%'
   ❌  WHERE "Material Type" = 'SENSORS'

3. "Plant" is INTEGER — filter without string quotes:
   ✅  WHERE "Plant" = 2001
   ❌  WHERE "Plant" = '2001'

4. Default row limit is 20 unless user asks for more.

5. NEVER run INSERT, UPDATE, DELETE, DROP, or ALTER.

6. Every response must include:
   a) The SQL query used (in a code block)
   b) The result as a readable table or list
   c) A plain-English explanation of what the answer means

7. If zero rows returned → retry with broader LIKE before saying "no data".

8. ROUND all dollar values and ratios to 2 decimal places.
"""


# ---------------- AGENT ----------------

@st.cache_resource
def initialize_agent():
    load_excel_to_sqlite()

    engine = create_engine(f"sqlite:///{DB_PATH}")
    db = SQLDatabase(engine=engine)

    if "OPENAI_API_KEY" not in st.secrets:
        st.error("⚠️ Please add OPENAI_API_KEY in Streamlit secrets.")
        st.stop()

    api_key = st.secrets["OPENAI_API_KEY"]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    agent = create_react_agent(llm, tools, prompt=build_system_prompt())
    logger.info("Agent initialised successfully.")
    return agent, engine


# ---------------- SCHEMA EXPANDER ----------------

def show_schema_expander(engine):
    with st.expander("🔍 View Database Schema & Sample Data", expanded=False):
        try:
            df_preview = pd.read_sql("SELECT * FROM inventory LIMIT 10", engine)
            total = pd.read_sql("SELECT COUNT(*) AS cnt FROM inventory", engine)["cnt"][0]
            st.write(f"**Total rows: {total:,}**")
            st.write(f"**Columns ({len(df_preview.columns)}):** {list(df_preview.columns)}")
            st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load schema preview: {e}")


# ---------------- UI ----------------

def main():
    st.title("📦 Inventory NLQ Chatbot")
    st.markdown("Ask questions about your inventory data in plain English.")

    with st.sidebar:
        st.header("📊 Quick Questions")

        example_questions = [
            "What materials have the highest shelf stock?",
            "Show me all raw materials in plant 2001",
            "Which SOP families have the most demand?",
            "What is the total WIP value by material type?",
            "List materials where safety stock exceeds demand",
            "How many unique plants are there?",
            "Show top 10 materials by shelf stock value",
            "What are the distinct material types?",
            "Show shelf stock for SENSORS across all plants",
            "What is the DOH vs demand ratio by plant?",
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

    agent, engine = initialize_agent()

    show_schema_expander(engine)
    st.markdown("---")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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


# import os
# import logging
# import sys
# import streamlit as st
# import pandas as pd
# from sqlalchemy import create_engine

# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_openai import ChatOpenAI

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# from langgraph.prebuilt import create_react_agent


# # ---------------- CONFIG ----------------

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - [%(levelname)s] - %(message)s",
#     handlers=[
#         logging.FileHandler("app_log.txt", encoding="utf-8"),
#         logging.StreamHandler(sys.stderr),
#     ],
# )
# logger = logging.getLogger(__name__)

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "inventory.db")
# EXCEL_PATH = os.path.join(BASE_DIR,"Current Inventory.xlsx")  # ✅ FIXED

# # Streamlit page config
# st.set_page_config(
#     page_title="Inventory NLQ Chatbot",
#     page_icon="📦",
#     layout="wide",
# )

# # Session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "prefill_query" not in st.session_state:
#     st.session_state.prefill_query = None


# # ---------------- LOAD DATA ----------------

# def load_excel_to_sqlite():
#     """Load Excel into SQLite if DB doesn't exist."""
#     if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
#         logger.info("SQLite DB already exists, skipping load.")
#         return

#     if not os.path.exists(EXCEL_PATH):
#         st.error(f"❌ Excel file not found at: {EXCEL_PATH}")
#         st.stop()

#     logger.info(f"Loading Excel from {EXCEL_PATH}")
#     df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

#     engine = create_engine(f"sqlite:///{DB_PATH}")
#     df.to_sql("inventory", engine, if_exists="replace", index=False)
#     engine.dispose()

#     logger.info(f"Data written to {DB_PATH}")


# # ---------------- AGENT ----------------

# @st.cache_resource
# def initialize_agent():
#     load_excel_to_sqlite()

#     # DB
#     engine = create_engine(f"sqlite:///{DB_PATH}")
#     db = SQLDatabase(engine=engine)

#     # ✅ STREAMLIT SECRETS (IMPORTANT)
#     if "OPENAI_API_KEY" not in st.secrets:
#         st.error("⚠️ Please add OPENAI_API_KEY in Streamlit secrets.")
#         st.stop()

#     api_key = st.secrets["OPENAI_API_KEY"]

#     # LLM
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0,
#         api_key=api_key
#     )

#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#     tools = toolkit.get_tools()

#     system_prompt = """You are a helpful data analyst that answers questions about an inventory database using SQL.

# Rules:
# 1. First list tables and get the schema before writing queries.
# 2. Write correct SQLite queries.
# 3. Limit results to 20 rows unless user asks more.
# 4. NEVER run INSERT, UPDATE, DELETE, DROP.
# 5. Always explain results clearly.
# 6. If no results, say it clearly.
# 7. Always show the SQL query used.
# """

#     agent = create_react_agent(llm, tools, prompt=system_prompt)
#     return agent


# # ---------------- UI ----------------

# def main():
#     st.title("📦 Inventory NLQ Chatbot")
#     st.markdown("Ask questions about your inventory data in plain English.")

#     # Sidebar
#     with st.sidebar:
#         st.header("📊 Quick Questions")

#         example_questions = [
#             "What materials have the highest shelf stock?",
#             "Show me all raw materials in plant 2001",
#             "Which product families have the most demand?",
#             "What is the total WIP value by material type?",
#             "List materials where safety stock exceeds demand",
#             "How many unique plants are there?",
#         ]

#         for q in example_questions:
#             if st.button(q, key=q, use_container_width=True):
#                 st.session_state.prefill_query = q
#                 st.rerun()

#         st.markdown("---")

#         if st.button("🗑️ Clear Chat", use_container_width=True):
#             st.session_state.chat_history = []
#             st.session_state.prefill_query = None
#             st.rerun()

#     # Initialize agent
#     agent = initialize_agent()

#     # Show chat history
#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     # Input
#     user_input = st.chat_input("Ask a question about your inventory...")

#     if st.session_state.prefill_query:
#         user_input = st.session_state.prefill_query
#         st.session_state.prefill_query = None

#     if user_input:
#         st.session_state.chat_history.append({"role": "user", "content": user_input})

#         with st.chat_message("user"):
#             st.markdown(user_input)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     result = agent.invoke(
#                         {"messages": [{"role": "user", "content": user_input}]}
#                     )

#                     response = result["messages"][-1].content

#                     st.markdown(response)

#                     st.session_state.chat_history.append(
#                         {"role": "assistant", "content": response}
#                     )

#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}"
#                     logger.error(error_msg)

#                     st.error(error_msg)
#                     st.session_state.chat_history.append(
#                         {"role": "assistant", "content": error_msg}
#                     )


# if __name__ == "__main__":
#     main()
