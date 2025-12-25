"""
ADAPT-SQL Streamlit Application - Refactored
Cleaner version with display functions moved to separate module
"""
import streamlit as st
import json
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.adapt_baseline import ADAPTBaseline
from ui.enhanced_retry_engine import EnhancedRetryEngine
from ui.display_utils import (
    display_schema_tab,
    display_complexity_tab,
    display_examples_tab,
    display_routing_tab,
    display_sql_tab,
    display_validation_tab,
    display_execution_tab,
    display_evaluation_tab,
    display_retry_history_tab
)


st.set_page_config(
    page_title="ADAPT-SQL Pipeline (Enhanced)", 
    page_icon="SQL", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'spider_data' not in st.session_state:
    st.session_state.spider_data = None


def load_spider_data(json_path: str):
    """Load Spider dataset"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def get_schema_from_sqlite(db_path: str) -> dict:
    """Extract schema from SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_dict = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    'column_name': row[1],
                    'data_type': row[2],
                    'is_nullable': 'YES' if row[3] == 0 else 'NO'
                })
            schema_dict[table] = columns
        
        conn.close()
        return schema_dict
    except Exception as e:
        st.error(f"Error extracting schema: {e}")
        return {}


def get_foreign_keys_from_sqlite(db_path: str) -> list:
    """Extract foreign keys"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        foreign_keys = []
        for table in tables:
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            for row in cursor.fetchall():
                foreign_keys.append({
                    'from_table': table,
                    'from_column': row[3],
                    'to_table': row[2],
                    'to_column': row[4]
                })
        
        conn.close()
        return foreign_keys
    except Exception as e:
        st.error(f"Error extracting foreign keys: {e}")
        return []


def main():
    st.title("ADAPT-SQL Pipeline (Enhanced with Full Retry)")
    st.markdown("Complete Text-to-SQL with Automatic Retry on Execution/Evaluation Failures")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        model = st.selectbox("Model", ["llama3.2", "codellama", "mistral", "qwen2.5"])
        
        spider_json_path = st.text_input(
            "Spider dev.json",
            value="/home/smore123/ADAPT-SQL/data/spider/dev.json"
        )
        
        spider_db_dir = st.text_input(
            "Spider DB directory",
            value="/home/smore123/ADAPT-SQL/data/spider/spider_data/database"
        )
        
        vector_store_path = st.text_input(
            "Vector Store",
            value="./vector_store"
        )
        
        k_examples = st.slider("Similar Examples", 1, 20, 10)
        
        st.markdown("---")
        st.markdown("### Enhanced Retry Settings")
        
        enable_full_retry = st.checkbox("Enable Full Pipeline Retry", value=True)
        max_full_retries = st.slider("Max Full Retries", 0, 3, 2)
        min_eval_score = st.slider("Min Evaluation Score", 0.0, 1.0, 0.5, 0.1)
        
        st.markdown("---")
        
        if st.button("Load Dataset"):
            data = load_spider_data(spider_json_path)
            if data:
                st.session_state.spider_data = data
                st.success(f"Loaded {len(data)} examples")
        
        if st.session_state.spider_data:
            st.info(f"{len(st.session_state.spider_data)} examples loaded")
    
    # Main content
    if not st.session_state.spider_data:
        st.info("Load dataset from sidebar to begin")
        st.markdown("""
        ### How to use:
        1. Configure the paths in the sidebar
        2. Click "Load Dataset" to load Spider examples
        3. Select an example query to analyze
        4. Enable/disable enhanced retry as needed
        5. Click "Run Pipeline" to process
        
        ### Enhanced Retry Features:
        - **Automatic Full Retry**: If execution fails or evaluation score is low
        - **Feedback Context**: Uses errors from previous attempts to improve
        - **Multi-Attempt History**: View all retry attempts and their results
        - **Best Attempt Selection**: Automatically selects best result if retries exhausted
        """)
        return
    
    # Example selection
    st.header("Query Analysis")
    
    example_idx = st.selectbox(
        "Select Example",
        range(len(st.session_state.spider_data)),
        format_func=lambda i: f"#{i+1}: {st.session_state.spider_data[i]['question'][:70]}..."
    )
    
    example = st.session_state.spider_data[example_idx]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Question:**")
        st.info(example['question'])
    with col2:
        st.markdown("**Database:**")
        st.code(example['db_id'])
    
    if 'query' in example:
        with st.expander("Ground Truth SQL"):
            st.code(example['query'], language='sql')
    
    st.markdown("---")
    
    # Run pipeline button
    if st.button("Run Pipeline with Enhanced Retry", type="primary", use_container_width=True):
        with st.spinner("Processing with enhanced retry..."):
            db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
            
            if not db_path.exists():
                st.error("Database not found")
                return
            
            schema_dict = get_schema_from_sqlite(str(db_path))
            foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
            
            # Initialize ADAPT baseline
            adapt = ADAPTBaseline(model=model, vector_store_path=vector_store_path)
            
            gold_sql = example.get('query', None)
            
            if enable_full_retry:
                # Use enhanced retry engine
                retry_engine = EnhancedRetryEngine(
                    model=model,
                    max_full_retries=max_full_retries,
                    min_evaluation_score=min_eval_score
                )
                
                retry_result = retry_engine.run_with_full_retry(
                    adapt_baseline=adapt,
                    natural_query=example['question'],
                    schema_dict=schema_dict,
                    foreign_keys=foreign_keys,
                    k_examples=k_examples,
                    db_path=str(db_path),
                    gold_sql=gold_sql
                )
                
                result = retry_result['final_result']
                
                # Show retry summary
                st.success("Pipeline Complete with Enhanced Retry!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Attempts", retry_result['total_attempts'])
                with col2:
                    status = "Success" if retry_result['success'] else "Max Retries"
                    st.metric("Final Status", status)
                with col3:
                    if result.get('step11'):
                        st.metric("Final Score", f"{result['step11']['evaluation_score']:.2f}")
                
            else:
                # Normal pipeline without enhanced retry
                result = adapt.run_full_pipeline(
                    natural_query=example['question'],
                    schema_dict=schema_dict,
                    foreign_keys=foreign_keys,
                    k_examples=k_examples,
                    enable_retry=True,
                    db_path=str(db_path),
                    gold_sql=gold_sql,
                    enable_execution=True,
                    enable_evaluation=(gold_sql is not None)
                )
                
                retry_result = None
                st.success("Pipeline Complete!")
            
            st.markdown("---")
            
            # Display results in tabs
            if enable_full_retry and retry_result:
                tabs = [
                    "Schema", "Complexity", "Examples", 
                    "Route", "SQL", "Validation",
                    "Execution", "Evaluation", "Retry History"
                ]
            else:
                tabs = [
                    "Schema", "Complexity", "Examples",
                    "Route", "SQL", "Validation",
                    "Execution", "Evaluation"
                ]
            
            tab_objects = st.tabs(tabs)
            
            with tab_objects[0]:
                display_schema_tab(result)
            
            with tab_objects[1]:
                display_complexity_tab(result)
            
            with tab_objects[2]:
                display_examples_tab(result)
            
            with tab_objects[3]:
                display_routing_tab(result)
            
            with tab_objects[4]:
                display_sql_tab(result, example)
            
            with tab_objects[5]:
                display_validation_tab(result)
            
            with tab_objects[6]:
                display_execution_tab(result, example)
            
            with tab_objects[7]:
                display_evaluation_tab(result, example)
            
            if enable_full_retry and retry_result:
                with tab_objects[8]:
                    display_retry_history_tab(retry_result)


if __name__ == "__main__":
    main()