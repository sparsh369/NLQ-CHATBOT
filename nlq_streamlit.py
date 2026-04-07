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


# import os
# import logging
# import sys
# import streamlit as st
# import pandas as pd
# from sqlalchemy import create_engine, text

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
#     """Load Excel into SQLite with proper data cleaning."""
#     if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
#         logger.info("SQLite DB already exists, skipping load.")
#         return

#     if not os.path.exists(EXCEL_PATH):
#         st.error(f"❌ Excel file not found at: {EXCEL_PATH}")
#         st.stop()

#     logger.info(f"Loading Excel from {EXCEL_PATH}")
#     df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

#     # Strip trailing spaces from column names (CRITICAL FIX)
#     df.columns = [col.strip() for col in df.columns]
#     logger.info(f"Column names after strip: {list(df.columns)}")

#     # ===== DATA CLEANING =====
    
#     # 1. Replace empty strings with NULL for critical text columns
#     critical_cols = [
#         "Material Name", "SOP Family", "Product Family", 
#         "Material Type", "Product Group", "Material Application",
#         "Sub Application"
#     ]
#     for col in critical_cols:
#         if col in df.columns:
#             df[col] = df[col].replace('', None)
#             df[col] = df[col].replace(' ', None)
    
#     # 2. Fill numeric NULLs with 0 for calculation columns
#     numeric_cols = [
#         "Shelf Stock", "Shelf Stock ($)", "GIT", "GIT ($)", 
#         "WIP", "WIP($)", "DOH", "Safety Stock", "Demand"
#     ]
#     for col in numeric_cols:
#         if col in df.columns:
#             df[col] = df[col].fillna(0)
    
#     # 3. Remove rows with NULL Material Name (these are junk rows)
#     before_count = len(df)
#     df = df[df["Material Name"].notna()]
#     after_count = len(df)
#     logger.info(f"Removed {before_count - after_count} rows with NULL Material Name")
    
#     # 4. Log cleaning statistics
#     logger.info(f"Data cleaned: {len(df):,} valid rows retained")
#     logger.info(f"Rows with SOP Family: {df['SOP Family'].notna().sum():,} ({(df['SOP Family'].notna().sum() / len(df)) * 100:.1f}%)")
#     logger.info(f"Rows with Shelf Stock > 0: {(df['Shelf Stock'] > 0).sum():,}")
#     logger.info(f"Rows with Shelf Stock ($) > 0: {(df['Shelf Stock ($)'] > 0).sum():,}")
    
#     # ===== END CLEANING =====

#     engine = create_engine(f"sqlite:///{DB_PATH}")
#     df.to_sql("inventory", engine, if_exists="replace", index=False)
    
#     # Create indexes for better query performance
#     with engine.connect() as conn:
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_material_name ON inventory("Material Name")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_sop_family ON inventory("SOP Family")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_plant ON inventory("Plant")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_shelf_stock ON inventory("Shelf Stock ($)")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_material_type ON inventory("Material Type")'))
#         conn.execute(text('CREATE INDEX IF NOT EXISTS idx_product_family ON inventory("Product Family")'))
#         conn.commit()
    
#     engine.dispose()
#     logger.info(f"Data written to {DB_PATH} — {len(df):,} rows, {len(df.columns)} columns")


# # ---------------- ENHANCED SYSTEM PROMPT ----------------

# def build_system_prompt() -> str:
#     return """You are a precise inventory data analyst. You answer questions by writing and executing SQL
# queries against a SQLite database. Follow these rules EXACTLY to ensure accurate results.

# ════════════════════════════════════════════════════════
# DATABASE:  SQLite   TABLE: inventory   ROWS: ~126,000
# ════════════════════════════════════════════════════════

# ⚠️  CRITICAL RULES - FOLLOW THESE STRICTLY ⚠️

# 1. **ALWAYS USE "Material Name" COLUMN**
#    - Show "Material Name" (descriptive names), NEVER "Material" (codes)
#    - Exception: Only show "Material" if user explicitly asks for "material codes" or "material IDs"

# 2. **DEFAULT SORTING FOR "TOP" QUERIES**
#    - When user asks "top materials" WITHOUT explicit sorting criteria:
#      → ALWAYS sort by "Shelf Stock ($)" DESC (highest value first)
#    - Only sort by "Demand" if user explicitly mentions "demand" or "highest demand"
#    - Only sort by quantity if user explicitly asks for "quantity" or "units"

# 3. **AGGREGATION RULES**
#    - For dollar values across multiple materials: ALWAYS use "Shelf Stock ($)"
#    - NEVER sum "Shelf Stock" (quantities) across different materials (different UOMs)
#    - For counting: Use COUNT(DISTINCT "Material Name") for unique materials
#    - For filtering: Use "Shelf Stock ($)" > 0 for materials with value

# 4. **MANDATORY NULL FILTERS**
#    - Add "Material Name" IS NOT NULL to EVERY query showing materials
#    - Add "SOP Family" IS NOT NULL when filtering/grouping by SOP Family
#    - Add "Product Family" IS NOT NULL when filtering/grouping by Product Family
#    - Add "Product Group" IS NOT NULL when filtering/grouping by Product Group
#    - These filters are NOT optional - they prevent incorrect aggregations

# 5. **PRODUCT TYPE FILTERING**
#    - Use "SOP Family" column for product types (SENSORS, FIBER, NUHEAT, etc.)
#    - Use exact match (=) not LIKE for known SOP Family values
#    - NEVER use "MRP Controller Text" for product filtering (it contains planner names)

# 6. **NUMERIC PRECISION**
#    - ALWAYS use ROUND() for monetary values: ROUND(SUM("Shelf Stock ($)"), 2)
#    - Format percentages: ROUND((value / total) * 100, 2)
#    - Protect divisions: CASE WHEN denominator != 0 THEN numerator / denominator ELSE 0 END

# 7. **QUERY VALIDATION CHECKLIST**
#    Before executing, verify:
#    ✓ All column names wrapped in double-quotes
#    ✓ NULL filters added for categorical columns
#    ✓ Using "Shelf Stock ($)" for dollar aggregations
#    ✓ Using ROUND() for all monetary values
#    ✓ Correct sorting based on user intent
#    ✓ Showing "Material Name" not "Material"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN REFERENCE (wrap ALL column names in double-quotes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Column Name            | Type    | Description & Usage Notes
# -----------------------|---------|----------------------------------------------------
# "Plant"                | TEXT    | Plant/site ID (e.g., '2001', '2024')
# "Material"             | TEXT    | Material code (e.g., '363097-000') - DO NOT USE
# "Material Name"        | TEXT    | Full material name - ALWAYS USE THIS
# "Material Type"        | TEXT    | Category (Raw materials, Finished products, etc.)
# "UOM"                  | TEXT    | Unit of measure (FT, EA, KG, LB, etc.)
# "Shelf Stock"          | REAL    | Quantity (in UOM) - DO NOT SUM across materials
# "Shelf Stock ($)"      | REAL    | Dollar value - SAFE TO SUM (USE FOR AGGREGATIONS)
# "GIT"                  | REAL    | Goods in transit quantity
# "GIT ($)"              | REAL    | GIT dollar value
# "WIP"                  | REAL    | Work in progress quantity
# "WIP($)"               | REAL    | WIP dollar value
# "DOH"                  | REAL    | Days on hand
# "Safety Stock"         | REAL    | Minimum stock level
# "Demand"               | REAL    | Total demand quantity
# "Product Family"       | TEXT    | Product family code (ETL, HWAT, etc.)
# "SOP Family"           | TEXT    | PRIMARY product classification - use for filtering
# "Product Group"        | TEXT    | Detailed product group name
# "Material Group"       | TEXT    | Material grouping
# "Product Category"     | TEXT    | Category classification
# "Material Application" | TEXT    | Application type
# "Sub Application"      | TEXT    | Sub-application detail
# "ABC"                  | TEXT    | ABC classification (A, B, or C)
# "MRP Controller Text"  | TEXT    | Planner name - NOT a product category
# "Purchasing Group Text"| TEXT    | Purchasing group name

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICAL QUERY PATTERNS (COPY THESE EXACTLY)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 📌 PATTERN 1: TOP MATERIALS BY SHELF STOCK VALUE (DEFAULT)
# Use this when user asks "top materials" without specifying sorting:

# SELECT 
#     "Material Name",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Shelf Stock Value ($)",
#     ROUND(SUM("Demand"), 2) AS "Total Demand"
# FROM inventory
# WHERE "Material Name" IS NOT NULL
# GROUP BY "Material Name"
# ORDER BY SUM("Shelf Stock ($)") DESC
# LIMIT 10;

# 📌 PATTERN 2: FILTERING BY SOP FAMILY
# Always use exact match and NULL filter:

# SELECT 
#     "Plant",
#     COUNT(DISTINCT "Material Name") AS "Unique Materials",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Value ($)"
# FROM inventory
# WHERE "SOP Family" = 'SENSORS'
#   AND "SOP Family" IS NOT NULL
#   AND "Material Name" IS NOT NULL
#   AND "Shelf Stock ($)" > 0
# GROUP BY "Plant"
# ORDER BY SUM("Shelf Stock ($)") DESC;

# 📌 PATTERN 3: AGGREGATION BY CATEGORY

# SELECT 
#     "SOP Family",
#     COUNT(DISTINCT "Material Name") AS "Material Count",
#     ROUND(SUM("Shelf Stock ($)"), 2) AS "Total Value ($)",
#     ROUND(SUM("Demand"), 2) AS "Total Demand"
# FROM inventory
# WHERE "SOP Family" IS NOT NULL
#   AND "Material Name" IS NOT NULL
# GROUP BY "SOP Family"
# ORDER BY SUM("Shelf Stock ($)") DESC
# LIMIT 10;

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESPONSE STRUCTURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# For EVERY response, follow this structure:

# 1. **Brief acknowledgment** (1 sentence)
# 2. **SQL query in code block**
# 3. **Present results** in a clean formatted table
# 4. **Summary** (1-2 sentences with key insights)

# Keep responses concise, accurate, and professional.
# """


# # ---------------- FIXED AGENT INITIALIZATION ----------------

# @st.cache_resource
# def initialize_agent():
#     load_excel_to_sqlite()

#     engine = create_engine(f"sqlite:///{DB_PATH}")
#     db = SQLDatabase(engine=engine)

#     if "OPENAI_API_KEY" not in st.secrets:
#         st.error("⚠️ Please add OPENAI_API_KEY in Streamlit secrets.")
#         st.stop()

#     api_key = st.secrets["OPENAI_API_KEY"]

#     # Create LLM with system message
#     llm = ChatOpenAI(
#         model="gpt-4o",
#         temperature=0,
#         api_key=api_key,
#         max_tokens=2000,
#     )

#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#     tools = toolkit.get_tools()

#     # FIXED: Use prompt parameter instead of state_modifier
#     # The create_react_agent expects a string prompt, not state_modifier
#     agent = create_react_agent(
#         llm, 
#         tools,
#         # Pass system prompt directly as messages
#     )
    
#     logger.info("Agent initialized successfully with enhanced prompt.")
#     return agent, engine, build_system_prompt()


# # ---------------- SCHEMA EXPANDER WITH MORE STATS ----------------

# def show_schema_expander(engine):
#     with st.expander("🔍 View Database Schema & Sample Data", expanded=False):
#         try:
#             # Get comprehensive stats
#             stats = pd.read_sql("""
#                 SELECT 
#                     COUNT(*) as total_rows,
#                     COUNT(DISTINCT "Material Name") as unique_materials,
#                     COUNT(CASE WHEN "SOP Family" IS NOT NULL THEN 1 END) as rows_with_sop_family,
#                     COUNT(CASE WHEN "Shelf Stock ($)" > 0 THEN 1 END) as rows_with_shelf_stock,
#                     ROUND(SUM("Shelf Stock ($)"), 2) as total_shelf_stock_value,
#                     COUNT(DISTINCT "Plant") as unique_plants,
#                     COUNT(DISTINCT "Material Type") as unique_material_types
#                 FROM inventory
#             """, engine)
            
#             total = stats['total_rows'][0]
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Total Rows", f"{total:,}")
#                 st.metric("Unique Materials", f"{stats['unique_materials'][0]:,}")
#             with col2:
#                 st.metric("Rows with SOP Family", 
#                          f"{stats['rows_with_sop_family'][0]:,}",
#                          delta=f"{(stats['rows_with_sop_family'][0] / total) * 100:.1f}%")
#                 st.metric("Rows with Shelf Stock", 
#                          f"{stats['rows_with_shelf_stock'][0]:,}",
#                          delta=f"{(stats['rows_with_shelf_stock'][0] / total) * 100:.1f}%")
#             with col3:
#                 st.metric("Total Shelf Stock Value", f"${stats['total_shelf_stock_value'][0]:,.2f}")
#                 st.metric("Unique Plants", f"{stats['unique_plants'][0]}")
            
#             st.markdown("---")
#             st.subheader("Sample Data (First 10 rows)")
#             df_preview = pd.read_sql("SELECT * FROM inventory LIMIT 10", engine)
#             st.dataframe(df_preview, use_container_width=True)
            
#             st.markdown("---")
#             st.subheader("Column Overview")
#             st.write(f"**Total Columns:** {len(df_preview.columns)}")
#             st.write(f"**Columns:** {', '.join(df_preview.columns)}")
            
#         except Exception as e:
#             st.warning(f"Could not load schema preview: {e}")


# # ---------------- QUERY VALIDATOR ----------------

# def validate_and_log_query(user_query: str, sql_query: str = None, result: any = None):
#     """Log queries for debugging and validation"""
#     logger.info(f"\n{'='*80}")
#     logger.info(f"USER QUERY: {user_query}")
#     if sql_query:
#         logger.info(f"SQL GENERATED:\n{sql_query}")
#     if result:
#         logger.info(f"RESULT: {result}")
#     logger.info(f"{'='*80}\n")


# # ---------------- ENHANCED UI ----------------

# def main():
#     st.title("📦 Inventory NLQ Chatbot")
#     st.markdown("Ask questions about your inventory data in plain English.")
    
#     # Add tips section
#     with st.expander("💡 Tips for Better Results", expanded=False):
#         st.markdown("""
#         **For accurate results:**
#         - ✅ Be specific: "Show top 10 materials by shelf stock value"
#         - ✅ Use exact terms: "SENSORS", "NUHEAT", "Raw materials"
#         - ✅ Specify sorting: "by value", "by demand", "by quantity"
        
#         **Examples of good questions:**
#         - "What are the top 10 materials by shelf stock value?"
#         - "Show total shelf stock value for SENSORS by plant"
#         - "How many unique materials are in the NUHEAT family?"
#         - "List raw materials with shelf stock greater than $10,000"
#         """)

#     with st.sidebar:
#         st.header("📊 Quick Questions")

#         example_questions = [
#             "Show top 10 materials by shelf stock value",
#             "What materials have the highest shelf stock value?",
#             "Show shelf stock for SENSORS across all plants",
#             "Which SOP families have the most shelf stock value?",
#             "Show demand vs shelf stock for top 10 materials",
#             "List all raw materials with shelf stock > 0",
#             "What is the total shelf stock value by material type?",
#             "Show top materials by demand",
#             "How many unique materials are in the NUHEAT family?",
#             "What plants have FIBER products in stock?",
#             "Show ABC classification breakdown by value",
#             "List top 5 plants by total inventory value",
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

#     agent, engine, system_prompt = initialize_agent()

#     show_schema_expander(engine)
#     st.markdown("---")

#     # Display chat history
#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     # Handle user input
#     user_input = st.chat_input("Ask a question about your inventory...")

#     if st.session_state.prefill_query:
#         user_input = st.session_state.prefill_query
#         st.session_state.prefill_query = None

#     if user_input:
#         st.session_state.chat_history.append({"role": "user", "content": user_input})

#         with st.chat_message("user"):
#             st.markdown(user_input)

#         with st.chat_message("assistant"):
#             with st.spinner("🤔 Analyzing your query..."):
#                 try:
#                     # Log the query for debugging
#                     validate_and_log_query(user_input)
                    
#                     # Prepend system prompt to the user message
#                     full_message = f"{system_prompt}\n\nUser Question: {user_input}"
                    
#                     # Invoke agent with enhanced error handling
#                     result = agent.invoke(
#                         {"messages": [{"role": "user", "content": full_message}]}
#                     )
                    
#                     response = result["messages"][-1].content
                    
#                     # Log the response
#                     validate_and_log_query(user_input, result=response)
                    
#                     st.markdown(response)
#                     st.session_state.chat_history.append(
#                         {"role": "assistant", "content": response}
#                     )
                    
#                 except Exception as e:
#                     error_msg = f"❌ **Error occurred:** {str(e)}\n\n"
#                     error_msg += "**Troubleshooting tips:**\n"
#                     error_msg += "- Check if your query is specific enough\n"
#                     error_msg += "- Try rephrasing using exact column names\n"
#                     error_msg += "- Verify the data exists in the database\n"
#                     error_msg += "- Check the logs in `app_log.txt` for details"
                    
#                     logger.error(f"Error processing query: {user_input}")
#                     logger.error(f"Error details: {str(e)}", exc_info=True)
                    
#                     st.error(error_msg)
#                     st.session_state.chat_history.append(
#                         {"role": "assistant", "content": error_msg}
#                     )


# if __name__ == "__main__":
#     main()

