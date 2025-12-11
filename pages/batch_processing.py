"""
ADAPT-SQL Batch Processing Page - Refactored
Process multiple queries with execution, evaluation, and retry support
"""
import streamlit as st
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from adapt_baseline import ADAPTBaseline
from enhanced_retry_engine import EnhancedRetryEngine
from batch_utils import (
    display_batch_summary,
    display_complexity_distribution,
    display_execution_summary,
    display_evaluation_summary,
    display_retry_summary,
    display_query_summary_card,
    display_query_details,
    filter_results,
    export_summary_csv,
    export_full_json,
    display_error_analysis
)


st.set_page_config(
    page_title="Batch Processing - ADAPT-SQL", 
    page_icon="Batch", 
    layout="wide"
)


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
    st.title("Batch Processing")
    st.markdown("Process multiple queries and analyze results in detail")
    st.markdown("---")
    
    # Configuration sidebar
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
        st.markdown("### Batch Settings")
        
        num_queries = st.number_input("Number of Queries", min_value=1, max_value=1000, value=50)
        start_idx = st.number_input("Start Index", min_value=0, value=0)
        
        st.markdown("---")
        st.markdown("### Processing Options")
        
        enable_validation_retry = st.checkbox("Enable Validation Retry", value=True)
        enable_execution = st.checkbox("Enable SQL Execution", value=True)
        enable_evaluation = st.checkbox("Enable Evaluation", value=True)
        enable_full_retry = st.checkbox("Enable Full Pipeline Retry", value=False)
        
        if enable_full_retry:
            max_full_retries = st.slider("Max Full Retries", 0, 3, 1)
            min_eval_score = st.slider("Min Evaluation Score", 0.0, 1.0, 0.5, 0.1)
        
        st.markdown("---")
        
        if st.button("Load Dataset"):
            data = load_spider_data(spider_json_path)
            if data:
                st.session_state.spider_data = data
                st.success(f"Loaded {len(data)} examples")
        
        if 'spider_data' in st.session_state and st.session_state.spider_data:
            st.info(f"{len(st.session_state.spider_data)} examples loaded")
    
    # Main content
    if 'spider_data' not in st.session_state or not st.session_state.spider_data:
        st.info("Load dataset from sidebar to begin")
        st.markdown("""
        ### How to use:
        1. Configure the paths in the sidebar
        2. Click "Load Dataset" to load Spider examples
        3. Set the number of queries to process and starting index
        4. Enable/disable execution, evaluation, and retry options
        5. Click "Run Batch Processing" to process multiple queries
        6. View detailed results for each query
        7. Export results as CSV or JSON
        
        ### Processing Options:
        - **Validation Retry**: Attempts to fix validation errors (Step 8)
        - **SQL Execution**: Executes generated SQL on database (Step 10)
        - **Evaluation**: Compares results with ground truth (Step 11)
        - **Full Pipeline Retry**: Retries entire pipeline if execution fails or score is low
        """)
        return
    
    # Batch processing section
    st.markdown("## Batch Processing")
    
    end_idx = min(start_idx + num_queries, len(st.session_state.spider_data))
    st.info(f"Will process queries {start_idx} to {end_idx - 1} ({end_idx - start_idx} queries)")
    
    # Display processing options summary
    options = []
    if enable_validation_retry:
        options.append("Validation Retry")
    if enable_execution:
        options.append("Execution")
    if enable_evaluation:
        options.append("Evaluation")
    if enable_full_retry:
        options.append(f"Full Retry (max {max_full_retries})")
    
    if options:
        st.caption(f"Enabled: {', '.join(options)}")
    
    if st.button("Run Batch Processing", type="primary", use_container_width=True):
        # Initialize ADAPT
        adapt = ADAPTBaseline(model=model, vector_store_path=vector_store_path)
        
        # Initialize retry engine if needed
        retry_engine = None
        if enable_full_retry:
            retry_engine = EnhancedRetryEngine(
                model=model,
                max_full_retries=max_full_retries,
                min_evaluation_score=min_eval_score
            )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        # Process each query
        for i in range(start_idx, end_idx):
            example = st.session_state.spider_data[i]
            
            status_text.text(f"Processing query {i - start_idx + 1}/{end_idx - start_idx}: {example['question'][:50]}...")
            
            # Get database path
            db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
            
            if not db_path.exists():
                st.warning(f"Database not found for query {i}: {example['db_id']}")
                continue
            
            # Load schema
            schema_dict = get_schema_from_sqlite(str(db_path))
            foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
            
            gold_sql = example.get('query', None)
            
            # Run pipeline
            try:
                if enable_full_retry and retry_engine:
                    # Use enhanced retry engine
                    retry_result = retry_engine.run_with_full_retry(
                        adapt_baseline=adapt,
                        natural_query=example['question'],
                        schema_dict=schema_dict,
                        foreign_keys=foreign_keys,
                        k_examples=k_examples,
                        db_path=str(db_path) if enable_execution else None,
                        gold_sql=gold_sql if enable_evaluation else None
                    )
                    result = retry_result['final_result']
                    result['retry_info'] = {
                        'total_attempts': retry_result['total_attempts'],
                        'success': retry_result['success']
                    }
                    
                    # Store full retry result for detailed view
                    results.append({
                        'index': i,
                        'example': example,
                        'result': result,
                        'retry_result': retry_result  # Store full retry result
                    })
                else:
                    # Normal pipeline
                    result = adapt.run_full_pipeline(
                        example['question'],
                        schema_dict,
                        foreign_keys,
                        k_examples=k_examples,
                        enable_retry=enable_validation_retry,
                        db_path=str(db_path) if enable_execution else None,
                        gold_sql=gold_sql if enable_evaluation else None,
                        enable_execution=enable_execution,
                        enable_evaluation=enable_evaluation
                    )
                    
                    results.append({
                        'index': i,
                        'example': example,
                        'result': result,
                        'retry_result': None
                    })
            except Exception as e:
                st.error(f"Error processing query {i}: {e}")
            
            # Update progress
            progress_bar.progress((i - start_idx + 1) / (end_idx - start_idx))
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.batch_results = results
        st.session_state.batch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        st.success(f"Batch processing complete! Processed {len(results)} queries.")
    
    # Display results if available
    if 'batch_results' in st.session_state:
        st.markdown("---")
        st.markdown(f"## Results (Generated at {st.session_state.batch_timestamp})")
        
        results = st.session_state.batch_results
        
        # Summary statistics
        display_batch_summary(results)
        
        st.markdown("---")
        
        # Complexity distribution
        display_complexity_distribution(results)
        
        st.markdown("---")
        
        # Execution summary (if enabled)
        if any(r['result'].get('step10_generated') for r in results):
            display_execution_summary(results)
            st.markdown("---")
        
        # Evaluation summary (if enabled)
        if any(r['result'].get('step11') for r in results):
            display_evaluation_summary(results)
            st.markdown("---")
        
        # Retry summary (if enabled)
        if any(r['result'].get('retry_info') for r in results):
            display_retry_summary(results)
            st.markdown("---")
        
        # Error analysis
        display_error_analysis(results)
        
        st.markdown("---")
        
        # Detailed results with filtering
        st.markdown("### Detailed Query Results")
        
        # Filter options
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filter_complexity = st.multiselect(
                "Filter by Complexity",
                ["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"],
                default=["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"]
            )
        
        with col2:
            filter_validity = st.selectbox(
                "Filter by Validity",
                ["All", "Valid Only", "Invalid Only"],
                index=0
            )
        
        with col3:
            filter_execution = st.selectbox(
                "Filter by Execution",
                ["All", "Success Only", "Failed Only"],
                index=0
            )
        
        with col4:
            filter_evaluation = st.selectbox(
                "Filter by Evaluation",
                ["All", "High Score (>=0.7)", "Low Score (<0.5)"],
                index=0
            )
        
        # Apply filters
        filtered_results = filter_results(
            results, 
            filter_complexity, 
            filter_validity,
            filter_execution,
            filter_evaluation
        )
        
        st.info(f"Showing {len(filtered_results)} of {len(results)} queries")
        
        # Display view options
        view_mode = st.radio(
            "View Mode",
            ["Summary Cards", "Detailed Expandable"],
            horizontal=True
        )
        
        # Display results based on view mode
        if view_mode == "Summary Cards":
            for r in filtered_results:
                display_query_summary_card(r['index'], r['example'], r['result'])
                st.markdown("---")
        else:
            for r in filtered_results:
                display_query_details(
                    r['index'], 
                    r['example'], 
                    r['result'],
                    r.get('retry_result')  # Pass retry result if available
                )
        
        # Export options
        st.markdown("---")
        st.markdown("### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Summary CSV"):
                csv = export_summary_csv(results)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"adapt_sql_batch_{st.session_state.batch_timestamp.replace(' ', '_').replace(':', '-')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Full JSON"):
                json_str = export_full_json(results)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"adapt_sql_batch_{st.session_state.batch_timestamp.replace(' ', '_').replace(':', '-')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()