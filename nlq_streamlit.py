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

#     # Strip trailing spaces from column names (e.g. 'Product Family ' -> 'Product Family')
#     df.columns = [col.strip() for col in df.columns]

#     engine = create_engine(f"sqlite:///{DB_PATH}")
#     df.to_sql("inventory", engine, if_exists="replace", index=False)
#     engine.dispose()
#     logger.info(f"Data written to {DB_PATH} — {len(df):,} rows, {len(df.columns)} columns")


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

    # Strip trailing spaces from column names
    df.columns = [col.strip() for col in df.columns]

    # ===== DATA CLEANING =====
    
    # 1. Replace empty strings with NULL for critical columns
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
    df = df[df["Material Name"].notna()]
    
    # 4. Log cleaning statistics
    logger.info(f"Data cleaned: {len(df):,} valid rows retained")
    
    # ===== END CLEANING =====

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("inventory", engine, if_exists="replace", index=False)
    engine.dispose()
    logger.info(f"Data written to {DB_PATH} — {len(df):,} rows, {len(df.columns)} columns")


def build_system_prompt() -> str:
    return """You are a helpful inventory data analyst. You answer questions by writing and running SQL
against a SQLite database. Think carefully before writing SQL — follow every rule below.

════════════════════════════════════════════════════════
DATABASE:  SQLite   TABLE: inventory   ROWS: 126,472
════════════════════════════════════════════════════════

⚠️  CRITICAL RULES - READ THESE FIRST ⚠️
1. ALWAYS use "Material Name" column (descriptive names), NEVER "Material" (codes)
2. ALWAYS filter: WHERE "Material Name" IS NOT NULL on EVERY query
3. When asked "top materials" → ALWAYS sort by "Shelf Stock ($)" DESC
4. ALWAYS add IS NOT NULL filters for: "SOP Family", "Product Family", "Product Group"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLUMN REFERENCE  (wrap EVERY column name in double-quotes in SQL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column Name            | Type    | What it means
-----------------------|---------|-----------------------------
"Plant"                | INTEGER | Plant/site ID e.g. 2001, 2024
"Material"             | TEXT    | Material code e.g. 363097-000 (DO NOT USE - use "Material Name" instead)
"Material Name"        | TEXT    | Full name of the material (ALWAYS USE THIS)
"Material Type"        | TEXT    | Category of material (see values below)
"UOM"                  | TEXT    | Unit of measure e.g. FT, EA, KG
"Shelf Stock"          | REAL    | Quantity sitting on shelf (in UOM units)
"Shelf Stock ($)"      | REAL    | Dollar value of shelf stock
"GIT"                  | REAL    | Goods in transit quantity
"GIT ($)"              | REAL    | Dollar value of GIT
"WIP"                  | REAL    | Work in progress quantity
"WIP($)"               | REAL    | Dollar value of WIP
"DOH"                  | REAL    | Days on hand
"Safety Stock"         | REAL    | Minimum stock to keep
"Demand"               | REAL    | Total demand quantity
"Product Family"       | TEXT    | Product family code e.g. ETL, HWAT
"SOP Family"           | TEXT    | SOP planning family — THE PRIMARY product classification column
"Product Group"        | TEXT    | Detailed product group name
"Material Group"       | TEXT    | Material group e.g. Custom Cable
"Product Category"     | TEXT    | Category e.g. PD / Project
"Material Application" | TEXT    | Application e.g. KA / Floor Heating
"Sub Application"      | TEXT    | Sub-application e.g. KSA / Leak Detection
"ABC"                  | TEXT    | ABC classification: A, B, or C
"MRP Controller Text"  | TEXT    | MRP controller/planner name — NOT a product category
"Purchasing Group Text"| TEXT    | Purchasing group name

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATERIAL DISPLAY RULE (CRITICAL - VIOLATION = WRONG ANSWER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ALWAYS ALWAYS ALWAYS use "Material Name" when displaying materials to users.
⚠️  NEVER use "Material" column unless explicitly asked for "material codes" or "material IDs".

EVERY QUERY showing materials MUST:
1. SELECT "Material Name" (not "Material")
2. Include WHERE "Material Name" IS NOT NULL
3. GROUP BY "Material Name" (if aggregating)

✅ CORRECT:
  SELECT "Material Name", 
         SUM("Shelf Stock ($)") AS total_value,
         SUM("Demand") AS total_demand
  FROM inventory
  WHERE "Material Name" IS NOT NULL
  GROUP BY "Material Name"
  ORDER BY total_value DESC
  LIMIT 10;

❌ WRONG (will show codes like 363097-000 instead of names):
  SELECT "Material", 
         SUM("Shelf Stock ($)") AS total_value
  FROM inventory
  GROUP BY "Material"

Exception: Only use "Material" if user explicitly requests:
- "Show material codes"
- "What are the material IDs"
- "List material numbers"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"TOP MATERIALS" QUERY RULE (CRITICAL - DEFINES SORTING ORDER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When user asks for "top materials" or "top N materials" WITHOUT explicit sorting:

DEFAULT SORT ORDER: "Shelf Stock ($)" DESC (highest shelf stock value first)

Examples triggering this rule:
- "Show demand vs shelf stock for top 10 materials"
- "Top 10 materials"
- "Show me the top materials"
- "List top 5 materials with demand and shelf stock"

✅ CORRECT - Sort by Shelf Stock ($):
  SELECT "Material Name",
         ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock Value ($)",
         ROUND(SUM("Demand"), 2) AS "Total Demand"
  FROM inventory
  WHERE "Material Name" IS NOT NULL
  GROUP BY "Material Name"
  ORDER BY SUM("Shelf Stock ($)") DESC  ← Sort by shelf stock
  LIMIT 10;

Only sort by "Demand" if user explicitly says:
- "top materials by demand"
- "materials with highest demand"
- "sort by demand"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NULL HANDLING - MANDATORY FILTERS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  These columns contain many NULL values and WILL pollute results if not filtered:

MANDATORY NULL FILTERS (add to EVERY relevant query):

1. "Material Name" IS NOT NULL 
   → Add to EVERY query showing materials (99% of queries)

2. "SOP Family" IS NOT NULL
   → Add when filtering/grouping by SOP Family

3. "Product Family" IS NOT NULL
   → Add when filtering/grouping by Product Family

4. "Product Group" IS NOT NULL
   → Add when filtering/grouping by Product Group

5. "Material Application" IS NOT NULL
   → Add when filtering/grouping by Material Application

6. "Sub Application" IS NOT NULL
   → Add when filtering/grouping by Sub Application

✅ CORRECT (with NULL filters):
  SELECT "SOP Family", 
         SUM("Shelf Stock ($)") AS total_value
  FROM inventory
  WHERE "SOP Family" IS NOT NULL        ← Required!
    AND "Material Name" IS NOT NULL     ← Required!
  GROUP BY "SOP Family"
  ORDER BY total_value DESC;

❌ WRONG (missing NULL filters - will include junk NULL rows):
  SELECT "SOP Family", 
         SUM("Shelf Stock ($)") AS total_value
  FROM inventory
  GROUP BY "SOP Family";

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWN COLUMN VALUES  (use exact match = for known values)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Material Type" — 11 exact values:
  Raw materials | Semifinished products | Finished products | Trading goods
  Packaging | Operating supplies-NON VA | Nonvaluated materials
  Prod. resources/tools | Optng suppl/Non Cos-VALUA | Spare parts | Services

"Plant" — 48 numeric IDs (sample):
  2001, 2006, 2007, 2012, 2013, 2014, 2015, 2018, 2019, 2020,
  2021, 2022, 2023, 2024, 2025, 2026, 3001, 3002, 3003 ...

"ABC":  A | B | C

"SOP Family" — EXACT known values (use = for these, not LIKE):
  MONO | MONO-CEL_D | SENSORS | FIBER-ZONE | RWC-BO | FIBER-COAT
  SENSORS ROPED CABLES | SEN-BULK | Reynosa Sensors | CMPT
  SENSORS SUB ASSY | Reynosa FrostGuards | NUHEAT | SEN-KITT
  nVent Thermal Europe | SENSORS SUB EPOXY | FIBER | PKG

  ⚠️  'SENSORS' in "SOP Family" means ONLY the exact value 'SENSORS'.
      It does NOT include 'SEN-BULK', 'SEN-KITT', 'SENSORS ROPED CABLES',
      'SENSORS SUB ASSY', 'SENSORS SUB EPOXY', or 'Reynosa Sensors'
      unless the user explicitly asks to include those families.

"Product Category" — 13 exact values:
  PD / Project | PD / Polymer Pipe Heat Tracing - BIS
  PD / Heat Tracing Components | PD / Polymer Pipe Heat Tracing - IND
  PD / Floor Heating | PD / Snow Melting & De-Icing
  PD / Control, Monitoring & Power Distribution
  PD / Fire and Performance Wiring | PD / Leak Detection
  PD / MI Heat Tracing | PD / Discountinued Products
  PD / Tip Clearance/Gadolina | PD / Mscellaneous

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRODUCT-TYPE FILTERING (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When filtering by product type (SENSORS, FIBER, NUHEAT, etc.):

✅ CORRECT - Use "SOP Family":
  WHERE "SOP Family" = 'SENSORS'
    AND "SOP Family" IS NOT NULL

❌ WRONG - Do NOT use "MRP Controller Text":
  WHERE "MRP Controller Text" LIKE '%SENSOR%'

"MRP Controller Text" contains planner names, NOT product types.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHELF STOCK - QUANTITY vs VALUE (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Shelf Stock" = RAW QUANTITY (FT, EA, LB, etc.) - different UOMs per row
"Shelf Stock ($)" = DOLLAR VALUE - safe to SUM across all materials

RULES:
1. When aggregating across multiple materials → ALWAYS use "Shelf Stock ($)"
2. Only use "Shelf Stock" when:
   - User explicitly asks for "quantity" or "units"
   - Filtering a single material
   - You also show the "UOM" column

3. Default interpretation of "shelf stock" = "Shelf Stock ($)"

✅ CORRECT:
  SELECT "Material Name",
         ROUND(SUM("Shelf Stock ($)"), 2) AS shelf_value
  FROM inventory
  WHERE "Material Name" IS NOT NULL
  GROUP BY "Material Name";

❌ WRONG (mixing FT + EA + LB):
  SELECT "Material Name",
         SUM("Shelf Stock") AS shelf_quantity
  FROM inventory
  GROUP BY "Material Name";

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZERO-STOCK FILTERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When user asks about "available stock" or "shelf stock":

ALWAYS add: AND "Shelf Stock ($)" > 0

This excludes items with no physical stock on shelves.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGGREGATION BY MATERIAL - STANDARD PATTERN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When aggregating metrics by material, use this exact pattern:

  SELECT "Material Name",
         ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock Value ($)",
         ROUND(SUM("Demand"), 2) AS "Total Demand"
  FROM inventory
  WHERE "Material Name" IS NOT NULL
  GROUP BY "Material Name"
  ORDER BY SUM("Shelf Stock ($)") DESC
  LIMIT 10;

Key points:
- Always SELECT "Material Name" (not "Material")
- Always GROUP BY "Material Name"
- Always include WHERE "Material Name" IS NOT NULL
- Use ROUND(..., 2) for all numeric results
- Label columns clearly with AS aliases

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CALCULATIONS & RATIOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For divisions, ALWAYS protect against divide-by-zero:

  ROUND(
    CASE WHEN SUM("Demand") = 0 OR SUM("Demand") IS NULL 
         THEN NULL
         ELSE SUM("DOH") / SUM("Demand")
    END, 2
  ) AS doh_demand_ratio

Rules:
- Wrap ALL divisions in CASE WHEN
- Filter IS NOT NULL on columns used in math
- ROUND all results to 2 decimal places
- Use meaningful column aliases

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL SQL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ALWAYS wrap column names in double-quotes:
   ✅ "Shelf Stock ($)"
   ❌ Shelf Stock ($)

2. Use = for exact matches, LIKE only for searches:
   ✅ WHERE "SOP Family" = 'SENSORS'
   ❌ WHERE "SOP Family" LIKE '%SENSORS%'

3. "Plant" is INTEGER (no quotes around values):
   ✅ WHERE "Plant" = 2001
   ❌ WHERE "Plant" = '2001'

4. Default LIMIT is 10 (not 20) unless user specifies

5. NEVER run INSERT, UPDATE, DELETE, DROP, or ALTER

6. Every response must include:
   a) SQL query in code block
   b) Results as a formatted table
   c) Plain-English summary

7. ROUND all dollar values and percentages to 2 decimals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Structure every response as:

1. Brief acknowledgment
2. SQL query in ```sql code block
3. Results in a clean table
4. 1-2 sentence summary of insights

Keep responses concise and professional.
"""
# ---------------- SYSTEM PROMPT ----------------

# def build_system_prompt() -> str:
#     return """You are a helpful inventory data analyst. You answer questions by writing and running SQL
# against a SQLite database. Think carefully before writing SQL — follow every rule below.

# ════════════════════════════════════════════════════════
# DATABASE:  SQLite   TABLE: inventory   ROWS: 126,472
# ════════════════════════════════════════════════════════

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN REFERENCE  (wrap EVERY column name in double-quotes in SQL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Column Name            | Type    | What it means
# -----------------------|---------|-----------------------------
# "Plant"                | INTEGER | Plant/site ID e.g. 2001, 2024
# "Material"             | TEXT    | Material code e.g. 363097-000
# "Material Name"        | TEXT    | Full name of the material
# "Material Type"        | TEXT    | Category of material (see values below)
# "UOM"                  | TEXT    | Unit of measure e.g. FT, EA, KG
# "Shelf Stock"          | REAL    | Quantity sitting on shelf (in UOM units)
# "Shelf Stock ($)"      | REAL    | Dollar value of shelf stock
# "GIT"                  | REAL    | Goods in transit quantity
# "GIT ($)"              | REAL    | Dollar value of GIT
# "WIP"                  | REAL    | Work in progress quantity
# "WIP($)"               | REAL    | Dollar value of WIP
# "DOH"                  | REAL    | Days on hand
# "Safety Stock"         | REAL    | Minimum stock to keep
# "Demand"               | REAL    | Total demand quantity
# "Product Family"       | TEXT    | Product family code e.g. ETL, HWAT
# "SOP Family"           | TEXT    | SOP planning family — THE PRIMARY product classification column
# "Product Group"        | TEXT    | Detailed product group name
# "Material Group"       | TEXT    | Material group e.g. Custom Cable
# "Product Category"     | TEXT    | Category e.g. PD / Project
# "Material Application" | TEXT    | Application e.g. KA / Floor Heating
# "Sub Application"      | TEXT    | Sub-application e.g. KSA / Leak Detection
# "ABC"                  | TEXT    | ABC classification: A, B, or C
# "MRP Controller Text"  | TEXT    | MRP controller/planner name — NOT a product category
# "Purchasing Group Text"| TEXT    | Purchasing group name

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KNOWN COLUMN VALUES  (use exact match = for known values, LIKE only when searching)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MATERIAL DISPLAY RULE (CRITICAL - ALWAYS FOLLOW THIS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# When showing material information to users, ALWAYS display the
# "Material Name" column (descriptive name), NOT the "Material"
# column (numeric code).

# CORRECT — showing material names:
#   SELECT "Material Name", "Shelf Stock ($)", "Demand"
#   FROM inventory
#   ORDER BY "Demand" DESC
#   LIMIT 10;

# WRONG — showing material codes (DO NOT DO THIS):
#   SELECT "Material", "Shelf Stock ($)", "Demand"
#   FROM inventory
#   ORDER BY "Demand" DESC
#   LIMIT 10;

# Exception: Only show "Material" code if the user explicitly asks
# for "material codes" or "material IDs". Otherwise ALWAYS use
# "Material Name" for better readability.

# When both are needed, show "Material Name" first, then "Material":
#   SELECT "Material Name", "Material", "Shelf Stock ($)"
#   FROM inventory
#   WHERE "Material" = '363097-000';
  
# "Material Type" — 11 exact values:
#   Raw materials | Semifinished products | Finished products | Trading goods
#   Packaging | Operating supplies-NON VA | Nonvaluated materials
#   Prod. resources/tools | Optng suppl/Non Cos-VALUA | Spare parts | Services

# "Plant" — 48 numeric IDs (sample):
#   2001, 2006, 2007, 2012, 2013, 2014, 2015, 2018, 2019, 2020,
#   2021, 2022, 2023, 2024, 2025, 2026, 3001, 3002, 3003 ...

# "ABC":  A | B | C

# "SOP Family" — EXACT known values (use = for these, not LIKE):
#   MONO | MONO-CEL_D | SENSORS | FIBER-ZONE | RWC-BO | FIBER-COAT
#   SENSORS ROPED CABLES | SEN-BULK | Reynosa Sensors | CMPT
#   SENSORS SUB ASSY | Reynosa FrostGuards | NUHEAT | SEN-KITT
#   nVent Thermal Europe | SENSORS SUB EPOXY | FIBER | PKG

#   ⚠️  'SENSORS' in "SOP Family" means ONLY the exact value 'SENSORS'.
#       It does NOT include 'SEN-BULK', 'SEN-KITT', 'SENSORS ROPED CABLES',
#       'SENSORS SUB ASSY', 'SENSORS SUB EPOXY', or 'Reynosa Sensors'
#       unless the user explicitly asks to include those families.

# "Product Family" — sample values:
#   ETL | XL-TRACE | HWAT | BTV | T2RED | ICESTOP | XTV | WGRD-H
#   XPI | CMPTS-IHTS | QTVR | WGRD-FS | EM | TT SENSORS | VPL
#   PLAB-SR | CCH | TRACETEK ACC/INSTR | JBS/JBM/T-100

# "Product Category" — 13 exact values:
#   PD / Project | PD / Polymer Pipe Heat Tracing - BIS
#   PD / Heat Tracing Components | PD / Polymer Pipe Heat Tracing - IND
#   PD / Floor Heating | PD / Snow Melting & De-Icing
#   PD / Control, Monitoring & Power Distribution
#   PD / Fire and Performance Wiring | PD / Leak Detection
#   PD / MI Heat Tracing | PD / Discountinued Products
#   PD / Tip Clearance/Gadolina | PD / Mscellaneous

# "Material Application" — 11 exact values:
#   KA / Commercial Heat-Tracing | KA / Industrial Heat-Tracing
#   KA / Floor Heating | KA / Speciality Heating
#   KA / Fire and Performance Wiring | KA / Leak Detection
#   KA / OFS | KA / Temperature Measurement | KA / Tip Clearance
#   KA / Rail and Transit Heating | KA / Gadolina

# "Sub Application" — 19 exact values:
#   KSA / Pipe Freeze Protection | KSA / Hot Water Temperature Maintenance
#   KSA / Industrial Heat-Tracing | KSA / Floor Heating
#   KSA / Roof & Gutter De-Icing | KSA / Commercial Components
#   KSA / Speciality Heating | KSA / Surface Snow Melting
#   KSA / Fire and Performance Wiring - BIS | KSA / Leak Detection
#   KSA / Fire and Performance Wiring - IND | KSA / Downhole/Bottomhole Heating
#   KSA / Project | KSA / In-Pipe Heating Cables
#   KSA / Temperature Measurement | KSA / Tip Clearance
#   KSA / Rail and Transit Heating | KSA / Oil Tank Freeze Protection
#   KSA / Gadolina

# "Material Group" — 271 unique values (sample):
#   Custom Cable | Resins - Engineering Plastics - General | Cables - General
#   Injection Molded - Plastic | Stamped Metal - Stamping | Rayclic
#   Electro-Mechanical - General | Electrical Supplies
#   Electronic Components - Connectors | Conductor ...

# "MRP Controller Text" — 141 unique values (sample):
#   SENSOR RAW-FIBER | FIBER SUBASSEMBLY | Mfg: I Plant, Elec
#   Buy: I Plant, Elec | Buy: Panel Compon | Imported Material
#   MPS Parts CN | SENSOR RAW-COMP | SENSOR WIP | INDUSTRIAL FG BO ...

#   ⚠️  "MRP Controller Text" contains PLANNER/CONTROLLER NAMES, not product
#       categories. Never use this column to filter by product type (e.g. sensors,
#       fiber). Always use "SOP Family" for product-type filtering.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORRECT COLUMN TO USE FOR PRODUCT-TYPE QUERIES  (CRITICAL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# When a user asks about a product type or product family such as
# "SENSORS", "FIBER", "NUHEAT" etc., ALWAYS filter on "SOP Family".

# NEVER filter on "MRP Controller Text" for product-type questions.
# "MRP Controller Text" is a person's name / planner code — it is NOT
# a reliable product classifier.

# CORRECT — filtering sensors by product classification:
#   WHERE "SOP Family" = 'SENSORS'

# WRONG — filtering sensors by planner name (DO NOT DO THIS):
#   WHERE "MRP Controller Text" LIKE '%SENSOR%'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NULL / MISSING DATA RULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTANT: Several columns are mostly NULL in this dataset:
#   "Product Family", "SOP Family", "Product Group",
#   "Material Application", "Sub Application"

# When filtering on these columns ALWAYS add:
#   AND "column_name" IS NOT NULL
# When aggregating these columns ALWAYS add:
#   WHERE "column_name" IS NOT NULL
# This prevents NULL rows from polluting your results.

# Example:
#   SELECT "SOP Family", SUM("Shelf Stock ($)") AS total_value
#   FROM inventory
#   WHERE "SOP Family" IS NOT NULL
#   GROUP BY "SOP Family"
#   ORDER BY total_value DESC
#   LIMIT 20;

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHELF STOCK QUANTITY vs VALUE RULE  (CRITICAL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# "Shelf Stock" stores RAW QUANTITY (FT, ST, LB, EA, etc.).
# Each row has a different UOM, so SUM("Shelf Stock") across
# multiple materials is MEANINGLESS — you'd be adding feet to
# pieces to pounds.

# "Shelf Stock ($)" stores the DOLLAR VALUE and IS safe to SUM
# across materials because it's always in the same unit ($).

# RULES:
# 1. For ANY aggregation across multiple materials or plants,
#    ALWAYS use SUM("Shelf Stock ($)") — never SUM("Shelf Stock").

# 2. Only use "Shelf Stock" (quantity) when:
#    - Filtering a single material with a known UOM
#    - The user explicitly asks for quantity/units (not value)
#    - You are also showing the UOM column alongside it

# 3. When user asks "shelf stock available", "total shelf stock",
#    "how much shelf stock" → default to "Shelf Stock ($)" and
#    label the result clearly as dollar value.

# 4. When a user asks for both quantity AND value, return both
#    columns separately — never combine them.

# EXCEPTION — explicit quantity request:
#   If the user or a defined business query explicitly asks for
#   shelf stock UNITS/QUANTITY (not dollar value), and the filter
#   already scopes to a single SOP Family or UOM group, then
#   SUM("Shelf Stock") is acceptable PROVIDED you also filter
#   "Shelf Stock" > 0 to exclude zero-stock rows.

#   Example of a valid quantity query:
#     SELECT "Plant", SUM("Shelf Stock") AS total_shelf_stock_units
#     FROM inventory
#     WHERE "SOP Family" = 'SENSORS'
#       AND "Shelf Stock" > 0
#       AND "SOP Family" IS NOT NULL
#     GROUP BY "Plant"
#     ORDER BY total_shelf_stock_units DESC;

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ZERO-STOCK FILTER RULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Many rows have "Shelf Stock" = 0 (items with no stock on hand).
# When the user asks about "available" stock or "shelf stock",
# ALWAYS add:
#   AND "Shelf Stock" > 0
# to exclude rows with no physical stock. This gives a true
# picture of what is actually available on the shelf.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STANDARD SENSORS QUERY TEMPLATE  (use this as the base)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Whenever a user asks about sensor shelf stock across plants,
# use this exact pattern:

#   SELECT "Plant",
#          COUNT("Material") AS item_count,
#          ROUND(SUM("Shelf Stock"), 2) AS total_shelf_stock_units,
#          ROUND(SUM("Shelf Stock ($)"), 2) AS total_shelf_stock_value
#   FROM inventory
#   WHERE "SOP Family" = 'SENSORS'
#     AND "Shelf Stock" > 0
#     AND "SOP Family" IS NOT NULL
#   GROUP BY "Plant"
#   ORDER BY total_shelf_stock_units DESC;

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPLEX CALCULATION RULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# For ratio / derived metrics, use SQLite-safe expressions:

# DOH vs Demand ratio by Plant:
#   SELECT "Plant",
#          ROUND(SUM("DOH"), 2)    AS total_DOH,
#          ROUND(SUM("Demand"), 2) AS total_Demand,
#          ROUND(
#            CASE WHEN SUM("Demand") = 0 THEN NULL
#                 ELSE SUM("DOH") / SUM("Demand")
#            END, 4
#          ) AS doh_demand_ratio
#   FROM inventory
#   WHERE "Demand" IS NOT NULL AND "DOH" IS NOT NULL
#   GROUP BY "Plant"
#   ORDER BY doh_demand_ratio DESC
#   LIMIT 20;

# Safety Stock coverage (Safety Stock / Demand):
#   ROUND(
#     CASE WHEN "Demand" = 0 OR "Demand" IS NULL THEN NULL
#          ELSE "Safety Stock" / "Demand"
#     END, 4
#   ) AS coverage_ratio

# Rules for calculations:
# - ALWAYS wrap divisions in a CASE WHEN denominator = 0 THEN NULL END
#   to prevent divide-by-zero crashes.
# - ALWAYS filter out NULLs with IS NOT NULL on columns used in math.
# - ALWAYS use ROUND(..., 2) on all dollar and ratio results.
# - For percentages multiply by 100 and label clearly.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPECIFIC MATERIAL LOOKUP RULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# When the user asks about a specific material code or name:
# 1. First try exact match on "Material":
#    WHERE "Material" = 'FIBER-XVR-32'
# 2. If zero rows → try LIKE on "Material":
#    WHERE "Material" LIKE '%FIBER-XVR%'
# 3. If still zero rows → try LIKE on "Material Name":
#    WHERE "Material Name" LIKE '%FIBER-XVR-32%'
# 4. If still zero → tell the user clearly that this material
#    does not exist in the database and suggest checking the spelling.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERAL SQL RULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. ALWAYS wrap column names in double-quotes.
#    ✅  SELECT "Shelf Stock ($)" FROM inventory
#    ❌  SELECT Shelf Stock ($) FROM inventory

# 2. For "SOP Family" and other classification columns with known exact
#    values, use = not LIKE:
#    ✅  WHERE "SOP Family" = 'SENSORS'
#    ❌  WHERE "SOP Family" LIKE '%SENSORS%'
#    Use LIKE only when doing a search across unknown values.

# 3. "Plant" is INTEGER — filter without string quotes:
#    ✅  WHERE "Plant" = 2001
#    ❌  WHERE "Plant" = '2001'

# 4. Default row limit is 20 unless user asks for more.

# 5. NEVER run INSERT, UPDATE, DELETE, DROP, or ALTER.

# 6. Every response must include:
#    a) The SQL query used (in a code block)
#    b) The result as a readable table or list
#    c) A plain-English explanation of what the answer means

# 7. If zero rows returned → retry with broader LIKE before saying "no data".

# 8. ROUND all dollar values and ratios to 2 decimal places.
# """


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

