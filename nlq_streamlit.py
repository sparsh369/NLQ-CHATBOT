# import os
# import logging
# import sys
# import streamlit as st
# import pandas as pd
# from sqlalchemy import create_engine, text
# import re

# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_openai import ChatOpenAI

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# from langgraph.prebuilt import create_react_agent


# # ---------------- CONFIG ----------------

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - [%(levelname)s] - %(message)s",
#     handlers=[
#         logging.FileHandler("app_log.txt", encoding="utf-8"),
#         logging.StreamHandler(sys.stderr),
#     ],
# )
# logger = logging.getLogger(__name__)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "inventory.db")
# EXCEL_PATH = os.path.join(BASE_DIR, "Current Inventory.xlsx")

# st.set_page_config(
#     page_title="Inventory NLQ Chatbot",
#     page_icon="📦",
#     layout="wide",
# )

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "prefill_query" not in st.session_state:
#     st.session_state.prefill_query = None


# # ---------------- LOAD DATA ----------------

# def load_excel_to_sqlite():
#     """Load Excel into SQLite with thorough data cleaning and type enforcement."""
#     if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
#         logger.info("SQLite DB already exists, skipping load.")
#         return

#     if not os.path.exists(EXCEL_PATH):
#         st.error(f"❌ Excel file not found at: {EXCEL_PATH}")
#         st.stop()

#     logger.info(f"Loading Excel from {EXCEL_PATH}")
#     df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

#     # Strip whitespace from column names
#     df.columns = [col.strip() for col in df.columns]
#     logger.info(f"Columns: {list(df.columns)}")

#     # ── 1. Remove the single summary "Total" row (Plant == 'Total') ──────────
#     before = len(df)
#     df = df[df["Plant"] != "Total"]
#     logger.info(f"Removed {before - len(df)} summary rows (Plant='Total')")

#     # ── 2. Remove rows with NULL Material Name ───────────────────────────────
#     before = len(df)
#     df = df[df["Material Name"].notna() & (df["Material Name"].str.strip() != "")]
#     logger.info(f"Removed {before - len(df)} rows with blank/null Material Name")

#     # ── 3. Clean text columns: empty string / whitespace → NULL ─────────────
#     text_cols = [
#         "Plant", "Material", "Material Name", "Material Type", "UOM",
#         "Product Family", "SOP Family", "Product Group", "Material Group",
#         "Product Category", "Material Application", "Sub Application",
#         "ABC", "MRP Controller Text", "Purchasing Group Text",
#     ]
#     for col in text_cols:
#         if col in df.columns:
#             df[col] = df[col].astype(str).str.strip()
#             df[col] = df[col].replace({"": None, "nan": None, "None": None})

#     # ── 4. CRITICAL: Numeric columns — fill NaN with 0 ──────────────────────
#     #      Demand has 114,956 NULLs meaning "no demand" — must become 0
#     #      Safety Stock, WIP, GIT all need the same treatment
#     numeric_cols = [
#         "Shelf Stock", "Shelf Stock ($)",
#         "GIT", "GIT ($)",
#         "WIP", "WIP($)",
#         "DOH", "Safety Stock", "Demand",
#     ]
#     for col in numeric_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

#     # ── 5. Log cleaning stats ────────────────────────────────────────────────
#     logger.info(f"Final row count: {len(df):,}")
#     logger.info(f"Shelf Stock ($) sum: ${df['Shelf Stock ($)'].sum():,.2f}")
#     logger.info(f"Demand nulls after fill: {df['Demand'].isna().sum()}")
#     logger.info(f"SOP Family non-null: {df['SOP Family'].notna().sum():,}")

#     # ── 6. Write to SQLite ───────────────────────────────────────────────────
#     engine = create_engine(f"sqlite:///{DB_PATH}")
#     df.to_sql("inventory", engine, if_exists="replace", index=False)

#     # ── 7. Indexes for performance ───────────────────────────────────────────
#     with engine.connect() as conn:
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_material_name   ON inventory("Material Name")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_sop_family      ON inventory("SOP Family")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_plant           ON inventory("Plant")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_shelf_stock_val ON inventory("Shelf Stock ($)")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_material_type   ON inventory("Material Type")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_product_family  ON inventory("Product Family")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_product_cat     ON inventory("Product Category")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_mrp_ctrl        ON inventory("MRP Controller Text")'))
#         conn.commit()

#     engine.dispose()
#     logger.info(f"DB written: {len(df):,} rows, {len(df.columns)} columns → {DB_PATH}")


# # ---------------- SYSTEM PROMPT ────────────────────────────────────────────

# def build_system_prompt() -> str:
#     return """You are a precise inventory data analyst. You answer questions ONLY by writing
# and executing SQL queries against a SQLite database called 'inventory'.

# ══════════════════════════════════════════════════════════════
# DATABASE: SQLite  |  TABLE: inventory  |  ROWS: ~126,000
# ══════════════════════════════════════════════════════════════

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — COLUMN REFERENCE  (ALWAYS wrap names in double-quotes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Column Name             SQLite Type  Notes
# ──────────────────────  ───────────  ──────────────────────────────────────────
# "Plant"                 TEXT         Plant/site ID: '2001','2006','7200', etc.
#                                      NOTE: The single summary row (Plant='Total')
#                                      has been removed — do NOT filter it out.
# "Material"              TEXT         Material code e.g. 'NGC40-C'. NEVER show this.
# "Material Name"         TEXT         Full descriptive name. ALWAYS show this.
# "Material Type"         TEXT         'Finished products','Trading goods',
#                                      'Raw materials','Semifinished products',
#                                      'Prod. resources/tools','Packaging',
#                                      'Operating supplies-NON VA',
#                                      'Optng suppl/Non Cos-VALUA',
#                                      'Nonvaluated materials','Spare parts','Services'
# "UOM"                   TEXT         Unit of measure: EA, FT, KG, LB, M, etc.
#                                      Differs per material — NEVER sum "Shelf Stock"
#                                      across materials.
# "Shelf Stock"           REAL         Quantity in UOM. DO NOT aggregate across
#                                      different materials (incompatible units).
# "Shelf Stock ($)"       REAL         Dollar value. SAFE to SUM/AVG/compare.
#                                      Raw float e.g. 86649592.7 — NOT a string.
#                                      Total across all data ≈ $173,299,185.
# "GIT"                   REAL         Goods-in-transit quantity.
# "GIT ($)"               REAL         Goods-in-transit dollar value.
# "WIP"                   REAL         Work-in-progress quantity. ~125k rows = 0.
# "WIP($)"                REAL         WIP dollar value. Can be negative.
# "DOH"                   REAL         Days on hand. Can be negative or very large.
# "Safety Stock"          REAL         Min stock level. 0 means no safety stock set.
#                                      NULLs were filled with 0 during load.
# "Demand"                REAL         ⚠️ CRITICAL: 114,956 rows were NULL (meaning
#                                      "no demand"). NULLs were filled to 0 at load.
#                                      Always treat 0 as "no demand".
# "Product Family"        TEXT         e.g. 'TLT','CMPTS-IHTS','XTV','BTV'. ~118k NULLs.
# "SOP Family"            TEXT         PRIMARY classification for product filtering.
#                                      Known values: 'RWC-BO','Reynosa Panel Shop',
#                                      'NUHEAT','SENSORS','MONO','FIBER-COAT',
#                                      'Reynosa Sensors','SEN-KITT','MONO-CEL_D',
#                                      'FIBER','FIBER-ZONE','CMPT','PKG',
#                                      'FIBER-SER','SEN-BULK','Reynosa FrostGuards',
#                                      'nVent Thermal Europe','Summit Australia Bid',
#                                      'SENSORS ROPED CABLES','SENSORS SUB ASSY'
#                                      ~114k rows have NULL SOP Family.
# "Product Group"         TEXT         Detailed group name. ~30k NULLs.
# "Material Group"        TEXT         Material grouping. ~84k NULLs.
# "Product Category"      TEXT         High-level category. Only 4 NULLs. Values:
#                                      'PD / Heat Tracing Components',
#                                      'PD / Project','PD / MI Heat Tracing',
#                                      'PD / Control, Monitoring & Power Distribution',
#                                      'PD / Snow Melting & De-Icing',
#                                      'PD / Fire and Performance Wiring',
#                                      'PD / Floor Heating',
#                                      'PD / Polymer Pipe Heat Tracing - IND',
#                                      'PD / Leak Detection',
#                                      'PD / Polymer Pipe Heat Tracing - BIS'
# "Material Application"  TEXT         Application type. ~30k NULLs.
# "Sub Application"       TEXT         Sub-application detail. ~30k NULLs.
# "ABC"                   TEXT         ABC class: 'A','B','C'. ~104k NULLs.
#                                      A=high value, B=medium, C=low.
# "MRP Controller Text"   TEXT         ⚠️ PLANNER NAMES ONLY — NOT a product category.
#                                      e.g. 'Planner 1','Miroslaw Nowak','Inv Other'.
#                                      ~95k NULLs. Use ONLY when user asks about
#                                      planners/MRP controllers specifically.
# "Purchasing Group Text" TEXT         Purchasing group name. ~98k NULLs.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — CRITICAL RULES  (non-negotiable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# RULE 1 — MATERIAL DISPLAY
#   ✅ ALWAYS show "Material Name" (descriptive name)
#   ❌ NEVER show "Material" (code) unless user explicitly asks for codes

# RULE 2 — DOLLAR AGGREGATION
#   ✅ SUM("Shelf Stock ($)")   ← dollar value, safe to aggregate
#   ❌ NEVER SUM("Shelf Stock") ← quantity, incompatible units across materials

# RULE 3 — DEMAND IS STORED AS 0 (NOT NULL)
#   NULLs in "Demand" were converted to 0 during data load.
#   ✅ WHERE "Demand" = 0          ← correct for "no demand"
#   ✅ WHERE "Demand" > 0          ← correct for "has demand"
#   ❌ WHERE "Demand" IS NULL      ← will return 0 rows (all NULLs are gone)
#   Safety Stock NULLs were also converted to 0.

# RULE 4 — NULL GUARDS FOR GROUPING COLUMNS
#   Add IS NOT NULL whenever you GROUP BY or filter on:
#   "SOP Family", "Product Family", "Product Group", "Material Group",
#   "ABC", "MRP Controller Text", "Purchasing Group Text"
#   (These columns have high NULL rates — grouping without filter skews results)

# RULE 5 — DEFAULT SORT
#   "Top materials" with no explicit sort → ORDER BY "Shelf Stock ($)" DESC
#   Only use "Demand" for sort if user says "demand", "most demanded", etc.

# RULE 6 — NUMERIC FORMATTING
#   ALWAYS: ROUND(SUM("Shelf Stock ($)"), 2)
#   Division: CASE WHEN denominator != 0 THEN num/denom ELSE 0 END
#   Percent:  ROUND(value * 100.0 / total, 2)

# RULE 7 — MRP CONTROLLER TEXT
#   ❌ NEVER use "MRP Controller Text" to identify product types
#   ✅ Use "SOP Family" or "Product Category" for product classification

# RULE 8 — SHELF STOCK ($) IS NUMERIC
#   "Shelf Stock ($)" stores raw floats like 86649592.7
#   ✅ Use directly in SUM(), AVG(), WHERE comparisons
#   ❌ Do NOT cast or treat as a string/formatted value

# RULE 9 — QUERY CHECKLIST (verify before every execution)
#   ✓ Column names in double-quotes
#   ✓ "Material Name" shown (not "Material")
#   ✓ Using "Shelf Stock ($)" for dollar sums
#   ✓ ROUND() on all monetary outputs
#   ✓ NULL guards on high-null categorical columns
#   ✓ Demand/Safety Stock compared as numbers (they are 0, not NULL)

# RULE 10 — RESPONSE FORMAT (every response must follow this)
#   1. One-sentence acknowledgment
#   2. SQL in a ```sql ... ``` code block  ← MANDATORY, always visible
#   3. Results in a clean table
#   4. One-sentence summary with key number

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — VERIFIED QUERY PATTERNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── PATTERN A: Total shelf stock value across all plants ─────────────────────
# -- Expected result: $173,299,185.40
# SELECT ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock Value ($)"
# FROM inventory
# WHERE "Material Name" IS NOT NULL;

# ── PATTERN B: Top materials by shelf stock value (default "top" query) ──────
# SELECT
#     "Material Name",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock ($)",
#     ROUND(SUM("Demand"), 2)          AS "Total Demand"
# FROM inventory
# WHERE "Material Name" IS NOT NULL
# GROUP BY "Material Name"
# ORDER BY SUM("Shelf Stock ($)") DESC
# LIMIT 10;

# ── PATTERN C: Overstock — Shelf Stock > Safety Stock, zero Demand ───────────
# -- Both Safety Stock and Demand are stored as 0 (not NULL) after data load.
# SELECT
#     COUNT(DISTINCT "Material Name") AS "Overstock Material Count"
# FROM inventory
# WHERE "Material Name" IS NOT NULL
#   AND "Shelf Stock" > "Safety Stock"
#   AND "Demand" = 0;

# -- With material detail:
# SELECT
#     "Material Name",
#     "UOM",
#     "Shelf Stock",
#     "Safety Stock",
#     "Demand",
#     ROUND("Shelf Stock ($)", 2) AS "Shelf Stock ($)"
# FROM inventory
# WHERE "Material Name" IS NOT NULL
#   AND "Shelf Stock" > "Safety Stock"
#   AND "Demand" = 0
# ORDER BY "Shelf Stock ($)" DESC
# LIMIT 20;

# ── PATTERN D: Filter by SOP Family ─────────────────────────────────────────
# SELECT
#     "Plant",
#     COUNT(DISTINCT "Material Name")  AS "Unique Materials",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Value ($)"
# FROM inventory
# WHERE "SOP Family" = 'SENSORS'
#   AND "SOP Family" IS NOT NULL
#   AND "Material Name" IS NOT NULL
# GROUP BY "Plant"
# ORDER BY SUM("Shelf Stock ($)") DESC;

# ── PATTERN E: Product Category breakdown ───────────────────────────────────
# SELECT
#     "Product Category",
#     COUNT(DISTINCT "Material Name")  AS "Material Count",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock ($)"
# FROM inventory
# WHERE "Product Category" IS NOT NULL
#   AND "Material Name" IS NOT NULL
# GROUP BY "Product Category"
# ORDER BY SUM("Shelf Stock ($)") DESC;

# ── PATTERN F: MRP Controller by inventory value ────────────────────────────
# -- "MRP Controller Text" = planner names, NOT product types
# SELECT
#     "MRP Controller Text",
#     COUNT(DISTINCT "Material Name")  AS "Material Count",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock ($)"
# FROM inventory
# WHERE "MRP Controller Text" IS NOT NULL
#   AND "Material Name" IS NOT NULL
# GROUP BY "MRP Controller Text"
# ORDER BY SUM("Shelf Stock ($)") DESC
# LIMIT 10;

# ── PATTERN G: ABC classification breakdown ─────────────────────────────────
# SELECT
#     "ABC",
#     COUNT(DISTINCT "Material Name")  AS "Material Count",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock ($)",
#     ROUND(SUM("Shelf Stock ($)") * 100.0 /
#           SUM(SUM("Shelf Stock ($)")) OVER (), 2) AS "% of Total"
# FROM inventory
# WHERE "ABC" IS NOT NULL
#   AND "Material Name" IS NOT NULL
# GROUP BY "ABC"
# ORDER BY "ABC";

# ── PATTERN H: SOP Family aggregation ───────────────────────────────────────
# SELECT
#     "SOP Family",
#     COUNT(DISTINCT "Material Name")  AS "Material Count",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Value ($)",
#     ROUND(SUM("Demand"), 2)          AS "Total Demand"
# FROM inventory
# WHERE "SOP Family" IS NOT NULL
#   AND "Material Name" IS NOT NULL
# GROUP BY "SOP Family"
# ORDER BY SUM("Shelf Stock ($)") DESC;

# ── PATTERN I: Material Type breakdown ──────────────────────────────────────
# SELECT
#     "Material Type",
#     COUNT(DISTINCT "Material Name")  AS "Unique Materials",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock ($)"
# FROM inventory
# WHERE "Material Type" IS NOT NULL
#   AND "Material Name" IS NOT NULL
# GROUP BY "Material Type"
# ORDER BY SUM("Shelf Stock ($)") DESC;

# ── PATTERN J: Top plants by total inventory value ──────────────────────────
# SELECT
#     "Plant",
#     COUNT(DISTINCT "Material Name")  AS "Unique Materials",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock ($)"
# FROM inventory
# WHERE "Plant" IS NOT NULL
#   AND "Material Name" IS NOT NULL
# GROUP BY "Plant"
# ORDER BY SUM("Shelf Stock ($)") DESC
# LIMIT 10;
# """


# # ---------------- SQL EXTRACTOR ─────────────────────────────────────────────

# def extract_sql_from_response(response_text: str) -> str:
#     """Extract the SQL query from the agent's markdown response."""
#     # Match ```sql ... ```
#     m = re.findall(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
#     if m:
#         return m[0].strip()
#     # Match ``` SELECT ... ```
#     m = re.findall(r'```\s*(SELECT.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
#     if m:
#         return m[0].strip()
#     # Bare SELECT block
#     m = re.findall(r'(SELECT\s+.+?)(?=\n\n|\Z)', response_text, re.DOTALL | re.IGNORECASE)
#     if m:
#         return m[0].strip().rstrip(';')
#     return None


# # ---------------- AGENT INIT ────────────────────────────────────────────────

# @st.cache_resource
# def initialize_agent():
#     load_excel_to_sqlite()

#     engine = create_engine(f"sqlite:///{DB_PATH}")
#     db = SQLDatabase(engine=engine)

#     if "OPENAI_API_KEY" not in st.secrets:
#         st.error("⚠️ Add OPENAI_API_KEY to Streamlit secrets.")
#         st.stop()

#     llm = ChatOpenAI(
#         model="gpt-4o",
#         temperature=0,
#         api_key=st.secrets["OPENAI_API_KEY"],
#         max_tokens=4000,
#     )

#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#     tools = toolkit.get_tools()
#     agent = create_react_agent(llm, tools)

#     logger.info("Agent initialized.")
#     return agent, engine, build_system_prompt()


# # ---------------- SCHEMA EXPANDER ───────────────────────────────────────────

# def show_schema_expander(engine):
#     with st.expander("🔍 Database Overview", expanded=False):
#         try:
#             stats = pd.read_sql("""
#                 SELECT
#                     COUNT(*)                                        AS total_rows,
#                     COUNT(DISTINCT "Material Name")                AS unique_materials,
#                     COUNT(DISTINCT "Plant")                        AS unique_plants,
#                     COUNT(CASE WHEN "SOP Family" IS NOT NULL THEN 1 END) AS rows_with_sop,
#                     COUNT(CASE WHEN "Shelf Stock ($)" > 0  THEN 1 END) AS rows_with_stock,
#                     ROUND(SUM("Shelf Stock ($)"), 2)               AS total_value,
#                     COUNT(CASE WHEN "Demand" > 0           THEN 1 END) AS rows_with_demand,
#                     COUNT(CASE WHEN "Shelf Stock" > "Safety Stock"
#                                AND "Demand" = 0            THEN 1 END) AS overstock_rows
#                 FROM inventory
#             """, engine)

#             total = stats["total_rows"][0]
#             c1, c2, c3, c4 = st.columns(4)
#             c1.metric("Total Rows",       f"{total:,}")
#             c1.metric("Unique Materials", f"{stats['unique_materials'][0]:,}")
#             c2.metric("Unique Plants",    f"{stats['unique_plants'][0]}")
#             c2.metric("Rows w/ SOP Family", f"{stats['rows_with_sop'][0]:,}")
#             c3.metric("Total Shelf Stock Value", f"${stats['total_value'][0]:,.2f}")
#             c3.metric("Rows w/ Stock > 0",  f"{stats['rows_with_stock'][0]:,}")
#             c4.metric("Rows w/ Demand > 0", f"{stats['rows_with_demand'][0]:,}")
#             c4.metric("Potential Overstock", f"{stats['overstock_rows'][0]:,}")

#             st.markdown("---")
#             st.subheader("Sample Data (10 rows)")
#             st.dataframe(
#                 pd.read_sql("SELECT * FROM inventory LIMIT 10", engine),
#                 use_container_width=True,
#             )

#             st.markdown("---")
#             col_info = pd.read_sql("""
#                 SELECT
#                     "SOP Family"      AS sop,
#                     "Material Type"   AS mat_type,
#                     "Product Category" AS prod_cat,
#                     "ABC"             AS abc
#                 FROM inventory LIMIT 1
#             """, engine)
#             st.subheader("Key Categorical Values")
#             t1, t2 = st.columns(2)
#             with t1:
#                 st.write("**SOP Family values:**")
#                 st.dataframe(
#                     pd.read_sql('SELECT "SOP Family", COUNT(*) as cnt FROM inventory WHERE "SOP Family" IS NOT NULL GROUP BY "SOP Family" ORDER BY cnt DESC', engine),
#                     use_container_width=True, height=200,
#                 )
#             with t2:
#                 st.write("**Material Type values:**")
#                 st.dataframe(
#                     pd.read_sql('SELECT "Material Type", COUNT(*) as cnt FROM inventory WHERE "Material Type" IS NOT NULL GROUP BY "Material Type" ORDER BY cnt DESC', engine),
#                     use_container_width=True, height=200,
#                 )

#         except Exception as e:
#             st.warning(f"Schema preview error: {e}")


# # ---------------- LOGGING HELPER ────────────────────────────────────────────

# def log_query(user_q, sql=None, result=None):
#     logger.info("=" * 80)
#     logger.info(f"USER: {user_q}")
#     if sql:
#         logger.info(f"SQL:\n{sql}")
#     if result:
#         logger.info(f"RESULT: {str(result)[:500]}")
#     logger.info("=" * 80)


# # ---------------- MAIN UI ───────────────────────────────────────────────────

# def main():
#     st.title("📦 Inventory NLQ Chatbot")
#     st.markdown("Ask questions about your inventory in plain English.")

#     with st.expander("💡 Example Questions", expanded=False):
#         st.markdown("""
# | Category | Example Question |
# |---|---|
# | Value | What is the total shelf stock value across all plants? |
# | Top items | What are the top 10 materials by shelf stock value? |
# | Overstock | How many materials have shelf stock > safety stock but zero demand? |
# | By category | Which product category has the highest shelf stock value? |
# | By MRP | Which MRP controller manages the most inventory value? |
# | By plant | Which plants have the most inventory? |
# | Filter | Show SENSORS shelf stock by plant |
# | ABC | Show ABC classification breakdown by value |
# | Material type | Total value by material type |
# | Demand | Show top 10 materials by demand |
#         """)

#     # ── Sidebar ──────────────────────────────────────────────────────────────
#     with st.sidebar:
#         st.header("📊 Quick Questions")
#         quick_qs = [
#             "What is the total shelf stock value across all plants?",
#             "Show top 10 materials by shelf stock value",
#             "How many materials have shelf stock > safety stock but zero demand?",
#             "Which product category has the highest shelf stock value?",
#             "Which MRP Controller Text has the most inventory value?",
#             "Show shelf stock for SENSORS across all plants",
#             "Which SOP families have the most shelf stock value?",
#             "Show ABC classification breakdown by total value",
#             "What is the total shelf stock value by material type?",
#             "List top 5 plants by total inventory value",
#             "Show demand vs shelf stock for top 10 materials",
#             "How many unique materials are in the NUHEAT SOP family?",
#         ]
#         for q in quick_qs:
#             if st.button(q, key=q, use_container_width=True):
#                 st.session_state.prefill_query = q
#                 st.rerun()

#         st.markdown("---")
#         if st.button("🗑️ Clear Chat", use_container_width=True):
#             st.session_state.chat_history = []
#             st.session_state.prefill_query = None
#             st.rerun()

#         st.markdown("---")
#         if st.button("🔄 Reload Data from Excel", use_container_width=True):
#             if os.path.exists(DB_PATH):
#                 os.remove(DB_PATH)
#             st.cache_resource.clear()
#             st.success("DB cleared — will reload on next query.")
#             st.rerun()

#     # ── Init ─────────────────────────────────────────────────────────────────
#     agent, engine, system_prompt = initialize_agent()
#     show_schema_expander(engine)
#     st.markdown("---")

#     # ── Chat history display ─────────────────────────────────────────────────
#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             if msg["role"] == "assistant":
#                 if msg.get("sql_query"):
#                     with st.expander("🔎 SQL Query", expanded=False):
#                         st.code(msg["sql_query"], language="sql")
#                 st.markdown(msg["content"])
#             else:
#                 st.markdown(msg["content"])

#     # ── Input ────────────────────────────────────────────────────────────────
#     user_input = st.chat_input("Ask a question about your inventory...")
#     if st.session_state.prefill_query:
#         user_input = st.session_state.prefill_query
#         st.session_state.prefill_query = None

#     if not user_input:
#         return

#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     with st.chat_message("assistant"):
#         with st.spinner("🤔 Analyzing..."):
#             try:
#                 log_query(user_input)

#                 full_message = f"{system_prompt}\n\nUser Question: {user_input}"
#                 result = agent.invoke(
#                     {"messages": [{"role": "user", "content": full_message}]}
#                 )
#                 response = result["messages"][-1].content
#                 sql_query = extract_sql_from_response(response)

#                 log_query(user_input, sql=sql_query, result=response)

#                 # Show SQL prominently
#                 if sql_query:
#                     with st.expander("🔎 Generated SQL Query", expanded=True):
#                         st.code(sql_query, language="sql")
#                 else:
#                     st.warning("⚠️ No SQL detected — the agent may have used a different approach.")

#                 st.markdown(response)

#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": response,
#                     "sql_query": sql_query,
#                 })

#             except Exception as e:
#                 err = (
#                     f"❌ **Error:** {str(e)}\n\n"
#                     "Try rephrasing your question or check `app_log.txt` for details."
#                 )
#                 logger.error(f"Query failed: {user_input} | {e}", exc_info=True)
#                 st.error(err)
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": err,
#                     "sql_query": None,
#                 })


# if __name__ == "__main__":
#     main()

import os
import logging
import sys
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import re

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
EXCEL_PATH = os.path.join(BASE_DIR, "Current Inventory.xlsx")

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
    """Load Excel into SQLite with proper data cleaning."""
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        logger.info("SQLite DB already exists, skipping load.")
        return

    if not os.path.exists(EXCEL_PATH):
        st.error(f"❌ Excel file not found at: {EXCEL_PATH}")
        st.stop()

    logger.info(f"Loading Excel from {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

    # Strip trailing spaces from column names (CRITICAL FIX)
    df.columns = [col.strip() for col in df.columns]
    logger.info(f"Column names after strip: {list(df.columns)}")

    # ===== DATA CLEANING =====
    
    # 1. Replace empty strings with NULL for critical text columns
    critical_cols = [
        "Material Name", "SOP Family", "Product Family", 
        "Material Type", "Product Group", "Material Application",
        "Sub Application"
    ]
    for col in critical_cols:
        if col in df.columns:
            df[col] = df[col].replace('', None)
            df[col] = df[col].replace(' ', None)
    
    # 2. Fill numeric NULLs with 0 for calculation columns
    numeric_cols = [
        "Shelf Stock", "Shelf Stock ($)", "GIT", "GIT ($)", 
        "WIP", "WIP($)", "DOH", "Safety Stock", "Demand"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 3. Remove rows with NULL Material Name (these are junk rows)
    before_count = len(df)
    df = df[df["Material Name"].notna()]
    after_count = len(df)
    logger.info(f"Removed {before_count - after_count} rows with NULL Material Name")
    
    # 4. Log cleaning statistics
    logger.info(f"Data cleaned: {len(df):,} valid rows retained")
    logger.info(f"Rows with SOP Family: {df['SOP Family'].notna().sum():,} ({(df['SOP Family'].notna().sum() / len(df)) * 100:.1f}%)")
    logger.info(f"Rows with Shelf Stock > 0: {(df['Shelf Stock'] > 0).sum():,}")
    logger.info(f"Rows with Shelf Stock ($) > 0: {(df['Shelf Stock ($)'] > 0).sum():,}")
    
    # ===== END CLEANING =====

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("inventory", engine, if_exists="replace", index=False)
    
    # Create indexes for better query performance
    with engine.connect() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_material_name ON inventory("Material Name")'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_sop_family ON inventory("SOP Family")'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_plant ON inventory("Plant")'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_shelf_stock ON inventory("Shelf Stock ($)")'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_material_type ON inventory("Material Type")'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_product_family ON inventory("Product Family")'))
        conn.commit()
    
    engine.dispose()
    logger.info(f"Data written to {DB_PATH} — {len(df):,} rows, {len(df.columns)} columns")


# ---------------- ENHANCED SYSTEM PROMPT ----------------

def build_system_prompt() -> str:
    return """You are a precise inventory data analyst. You answer questions by writing and executing SQL
queries against a SQLite database. Follow these rules EXACTLY to ensure accurate results.

════════════════════════════════════════════════════════
DATABASE:  SQLite   TABLE: inventory   ROWS: ~126,000
════════════════════════════════════════════════════════

⚠️  CRITICAL RULES - FOLLOW THESE STRICTLY ⚠️

1. **ALWAYS USE "Material Name" COLUMN**
   - Show "Material Name" (descriptive names), NEVER "Material" (codes)
   - Exception: Only show "Material" if user explicitly asks for "material codes" or "material IDs"

2. **DEFAULT SORTING FOR "TOP" QUERIES**
   - When user asks "top materials" WITHOUT explicit sorting criteria:
     → ALWAYS sort by "Shelf Stock ($)" DESC (highest value first)
   - Only sort by "Demand" if user explicitly mentions "demand" or "highest demand"
   - Only sort by quantity if user explicitly asks for "quantity" or "units"

3. **AGGREGATION RULES**
   - For dollar values across multiple materials: ALWAYS use "Shelf Stock ($)"
   - NEVER sum "Shelf Stock" (quantities) across different materials (different UOMs)
   - For counting: Use COUNT(DISTINCT "Material Name") for unique materials
   - For filtering: Use "Shelf Stock ($)" > 0 for materials with value

4. **MANDATORY NULL FILTERS**
   - Add "Material Name" IS NOT NULL to EVERY query showing materials
   - Add "SOP Family" IS NOT NULL when filtering/grouping by SOP Family
   - Add "Product Family" IS NOT NULL when filtering/grouping by Product Family
   - Add "Product Group" IS NOT NULL when filtering/grouping by Product Group
   - These filters are NOT optional - they prevent incorrect aggregations

5. **PRODUCT TYPE FILTERING**
   - Use "SOP Family" column for product types (SENSORS, FIBER, NUHEAT, etc.)
   - Use exact match (=) not LIKE for known SOP Family values
   - NEVER use "MRP Controller Text" for product filtering (it contains planner names)

6. **NUMERIC PRECISION**
   - ALWAYS use ROUND() for monetary values: ROUND(SUM("Shelf Stock ($)"), 2)
   - Format percentages: ROUND((value / total) * 100, 2)
   - Protect divisions: CASE WHEN denominator != 0 THEN numerator / denominator ELSE 0 END

7. **QUERY VALIDATION CHECKLIST**
   Before executing, verify:
   ✓ All column names wrapped in double-quotes
   ✓ NULL filters added for categorical columns
   ✓ Using "Shelf Stock ($)" for dollar aggregations
   ✓ Using ROUND() for all monetary values
   ✓ Correct sorting based on user intent
   ✓ Showing "Material Name" not "Material"

8. **RESPONSE FORMAT REQUIREMENT**
   ALWAYS include the SQL query in a markdown code block in your response.
   Format: ```sql\nYOUR QUERY HERE\n```
   The SQL query must be visible to the user in EVERY response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLUMN REFERENCE (wrap ALL column names in double-quotes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column Name            | Type    | Description & Usage Notes
-----------------------|---------|----------------------------------------------------
"Plant"                | TEXT    | Plant/site ID (e.g., '2001', '2024')
"Material"             | TEXT    | Material code (e.g., '363097-000') - DO NOT USE
"Material Name"        | TEXT    | Full material name - ALWAYS USE THIS
"Material Type"        | TEXT    | Category (Raw materials, Finished products, etc.)
"UOM"                  | TEXT    | Unit of measure (FT, EA, KG, LB, etc.)
"Shelf Stock"          | REAL    | Quantity (in UOM) - DO NOT SUM across materials
"Shelf Stock ($)"      | REAL    | Dollar value - SAFE TO SUM (USE FOR AGGREGATIONS)
"GIT"                  | REAL    | Goods in transit quantity
"GIT ($)"              | REAL    | GIT dollar value
"WIP"                  | REAL    | Work in progress quantity
"WIP($)"               | REAL    | WIP dollar value
"DOH"                  | REAL    | Days on hand
"Safety Stock"         | REAL    | Minimum stock level
"Demand"               | REAL    | Total demand quantity
"Product Family"       | TEXT    | Product family code (ETL, HWAT, etc.)
"SOP Family"           | TEXT    | PRIMARY product classification - use for filtering
"Product Group"        | TEXT    | Detailed product group name
"Material Group"       | TEXT    | Material grouping
"Product Category"     | TEXT    | Category classification
"Material Application" | TEXT    | Application type
"Sub Application"      | TEXT    | Sub-application detail
"ABC"                  | TEXT    | ABC classification (A, B, or C)
"MRP Controller Text"  | TEXT    | Planner name - NOT a product category
"Purchasing Group Text"| TEXT    | Purchasing group name

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL QUERY PATTERNS (COPY THESE EXACTLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 PATTERN 1: TOP MATERIALS BY SHELF STOCK VALUE (DEFAULT)
Use this when user asks "top materials" without specifying sorting:

SELECT 
    "Material Name",
    ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock Value ($)",
    ROUND(SUM("Demand"), 2) AS "Total Demand"
FROM inventory
WHERE "Material Name" IS NOT NULL
GROUP BY "Material Name"
ORDER BY SUM("Shelf Stock ($)") DESC
LIMIT 10;

📌 PATTERN 2: FILTERING BY SOP FAMILY
Always use exact match and NULL filter:

SELECT 
    "Plant",
    COUNT(DISTINCT "Material Name") AS "Unique Materials",
    ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Value ($)"
FROM inventory
WHERE "SOP Family" = 'SENSORS'
  AND "SOP Family" IS NOT NULL
  AND "Material Name" IS NOT NULL
  AND "Shelf Stock ($)" > 0
GROUP BY "Plant"
ORDER BY SUM("Shelf Stock ($)") DESC;

📌 PATTERN 3: AGGREGATION BY CATEGORY

SELECT 
    "SOP Family",
    COUNT(DISTINCT "Material Name") AS "Material Count",
    ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Value ($)",
    ROUND(SUM("Demand"), 2) AS "Total Demand"
FROM inventory
WHERE "SOP Family" IS NOT NULL
  AND "Material Name" IS NOT NULL
GROUP BY "SOP Family"
ORDER BY SUM("Shelf Stock ($)") DESC
LIMIT 10;

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For EVERY response, follow this structure:

1. **Brief acknowledgment** (1 sentence)
2. **SQL query in code block** (MANDATORY - must be visible)
   ```sql
   YOUR QUERY HERE
   ```
3. **Present results** in a clean formatted table
4. **Summary** (1-2 sentences with key insights)

Keep responses concise, accurate, and professional.
"""


# ---------------- SQL QUERY EXTRACTOR ----------------

def extract_sql_from_response(response_text: str) -> str:
    """
    Extract SQL query from the agent's response.
    Looks for SQL in code blocks or common SQL patterns.
    """
    # Pattern 1: SQL in markdown code blocks
    sql_block_pattern = r'```sql\s*(.*?)\s*```'
    matches = re.findall(sql_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    
    # Pattern 2: SQL in plain code blocks
    code_block_pattern = r'```\s*(SELECT.*?);?\s*```'
    matches = re.findall(code_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    
    # Pattern 3: Look for SELECT statement without code blocks
    select_pattern = r'(SELECT\s+.*?(?:;|\n\n|\Z))'
    matches = re.findall(select_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip().rstrip(';')
    
    return None


# ---------------- FIXED AGENT INITIALIZATION ----------------

@st.cache_resource
def initialize_agent():
    load_excel_to_sqlite()

    engine = create_engine(f"sqlite:///{DB_PATH}")
    db = SQLDatabase(engine=engine)

    if "OPENAI_API_KEY" not in st.secrets:
        st.error("⚠️ Please add OPENAI_API_KEY in Streamlit secrets.")
        st.stop()

    api_key = st.secrets["OPENAI_API_KEY"]

    # Create LLM with system message
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key,
        max_tokens=2000,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # FIXED: Use prompt parameter instead of state_modifier
    # The create_react_agent expects a string prompt, not state_modifier
    agent = create_react_agent(
        llm, 
        tools,
        # Pass system prompt directly as messages
    )
    
    logger.info("Agent initialized successfully with enhanced prompt.")
    return agent, engine, build_system_prompt()


# ---------------- SCHEMA EXPANDER WITH MORE STATS ----------------

def show_schema_expander(engine):
    with st.expander("🔍 View Database Schema & Sample Data", expanded=False):
        try:
            # Get comprehensive stats
            stats = pd.read_sql("""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT "Material Name") as unique_materials,
                    COUNT(CASE WHEN "SOP Family" IS NOT NULL THEN 1 END) as rows_with_sop_family,
                    COUNT(CASE WHEN "Shelf Stock ($)" > 0 THEN 1 END) as rows_with_shelf_stock,
                    ROUND(SUM("Shelf Stock ($)"), 2) as total_shelf_stock_value,
                    COUNT(DISTINCT "Plant") as unique_plants,
                    COUNT(DISTINCT "Material Type") as unique_material_types
                FROM inventory
            """, engine)
            
            total = stats['total_rows'][0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{total:,}")
                st.metric("Unique Materials", f"{stats['unique_materials'][0]:,}")
            with col2:
                st.metric("Rows with SOP Family", 
                         f"{stats['rows_with_sop_family'][0]:,}",
                         delta=f"{(stats['rows_with_sop_family'][0] / total) * 100:.1f}%")
                st.metric("Rows with Shelf Stock", 
                         f"{stats['rows_with_shelf_stock'][0]:,}",
                         delta=f"{(stats['rows_with_shelf_stock'][0] / total) * 100:.1f}%")
            with col3:
                st.metric("Total Shelf Stock Value", f"${stats['total_shelf_stock_value'][0]:,.2f}")
                st.metric("Unique Plants", f"{stats['unique_plants'][0]}")
            
            st.markdown("---")
            st.subheader("Sample Data (First 10 rows)")
            df_preview = pd.read_sql("SELECT * FROM inventory LIMIT 10", engine)
            st.dataframe(df_preview, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Column Overview")
            st.write(f"**Total Columns:** {len(df_preview.columns)}")
            st.write(f"**Columns:** {', '.join(df_preview.columns)}")
            
        except Exception as e:
            st.warning(f"Could not load schema preview: {e}")


# ---------------- QUERY VALIDATOR ----------------

def validate_and_log_query(user_query: str, sql_query: str = None, result: any = None):
    """Log queries for debugging and validation"""
    logger.info(f"\n{'='*80}")
    logger.info(f"USER QUERY: {user_query}")
    if sql_query:
        logger.info(f"SQL GENERATED:\n{sql_query}")
    if result:
        logger.info(f"RESULT: {result}")
    logger.info(f"{'='*80}\n")


# ---------------- ENHANCED UI ----------------

def main():
    st.title("📦 Inventory NLQ Chatbot")
    st.markdown("Ask questions about your inventory data in plain English.")
    
    # Add tips section
    with st.expander("💡 Tips for Better Results", expanded=False):
        st.markdown("""
        **For accurate results:**
        - ✅ Be specific: "Show top 10 materials by shelf stock value"
        - ✅ Use exact terms: "SENSORS", "NUHEAT", "Raw materials"
        - ✅ Specify sorting: "by value", "by demand", "by quantity"
        
        **Examples of good questions:**
        - "What are the top 10 materials by shelf stock value?"
        - "Show total shelf stock value for SENSORS by plant"
        - "How many unique materials are in the NUHEAT family?"
        - "List raw materials with shelf stock greater than $10,000"
        """)

    with st.sidebar:
        st.header("📊 Quick Questions")

        example_questions = [
            "Show top 10 materials by shelf stock value",
            "What materials have the highest shelf stock value?",
            "Show shelf stock for SENSORS across all plants",
            "Which SOP families have the most shelf stock value?",
            "Show demand vs shelf stock for top 10 materials",
            "List all raw materials with shelf stock > 0",
            "What is the total shelf stock value by material type?",
            "Show top materials by demand",
            "How many unique materials are in the NUHEAT family?",
            "What plants have FIBER products in stock?",
            "Show ABC classification breakdown by value",
            "List top 5 plants by total inventory value",
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

    agent, engine, system_prompt = initialize_agent()

    show_schema_expander(engine)
    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Extract and display SQL if available
                if "sql_query" in msg and msg["sql_query"]:
                    st.code(msg["sql_query"], language="sql")
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    # Handle user input
    user_input = st.chat_input("Ask a question about your inventory...")

    if st.session_state.prefill_query:
        user_input = st.session_state.prefill_query
        st.session_state.prefill_query = None

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query..."):
                try:
                    # Log the query for debugging
                    validate_and_log_query(user_input)
                    
                    # Prepend system prompt to the user message
                    full_message = f"{system_prompt}\n\nUser Question: {user_input}"
                    
                    # Invoke agent with enhanced error handling
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": full_message}]}
                    )
                    
                    response = result["messages"][-1].content
                    
                    # Extract SQL query from response
                    sql_query = extract_sql_from_response(response)
                    
                    # Log the response with SQL
                    validate_and_log_query(user_input, sql_query=sql_query, result=response)
                    
                    # Display SQL query prominently if found
                    if sql_query:
                        st.success("✅ **Generated SQL Query:**")
                        st.code(sql_query, language="sql")
                        st.markdown("---")
                    else:
                        st.warning("⚠️ No SQL query detected in response. The agent may have used a different approach.")
                    
                    # Display the response
                    st.markdown(response)
                    
                    # Save to chat history with SQL
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "sql_query": sql_query
                    })
                    
                except Exception as e:
                    error_msg = f"❌ **Error occurred:** {str(e)}\n\n"
                    error_msg += "**Troubleshooting tips:**\n"
                    error_msg += "- Check if your query is specific enough\n"
                    error_msg += "- Try rephrasing using exact column names\n"
                    error_msg += "- Verify the data exists in the database\n"
                    error_msg += "- Check the logs in `app_log.txt` for details"
                    
                    logger.error(f"Error processing query: {user_input}")
                    logger.error(f"Error details: {str(e)}", exc_info=True)
                    
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "sql_query": None
                    })


if __name__ == "__main__":
    main()



