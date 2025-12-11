"""
ADAPT-SQL Batch Processing Page with Incremental Saving
Process multiple queries with automatic checkpoints every 25 queries
"""
import streamlit as st
import json
import sqlite3
import pickle
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
    display_error_analysis,
    display_performance_breakdown,
    display_score_distribution,
    save_checkpoint,
    load_checkpoint,
    get_checkpoint_files
)


st.set_page_config(
    page_title="Batch Processing - ADAPT-SQL", 
    page_icon="📦", 
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
    st.title("📦 Batch Processing - ADAPT-SQL")
    st.markdown("Process multiple queries with automatic checkpoints every 25 queries")
    st.markdown("---")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model = st.selectbox("🤖 Model", ["llama3.2", "codellama", "mistral", "qwen2.5"])
        
        spider_json_path = st.text_input(
            "📄 Spider dev.json",
            value="/home/smore123/ADAPT-SQL/data/spider/dev.json"
        )
        
        spider_db_dir = st.text_input(
            "📁 Spider DB directory",
            value="/home/smore123/ADAPT-SQL/data/spider/spider_data/database"
        )
        
        vector_store_path = st.text_input(
            "🔍 Vector Store",
            value="./vector_store"
        )
        
        k_examples = st.slider("📚 Similar Examples", 1, 20, 10)
        
        st.markdown("---")
        st.markdown("### 🎯 Batch Settings")
        
        num_queries = st.number_input("Number of Queries", min_value=1, max_value=1000, value=10, step=5)
        start_idx = st.number_input("Start Index", min_value=0, value=0)
        
        checkpoint_interval = st.number_input("Checkpoint Interval", min_value=5, max_value=100, value=25, step=5)
        st.caption(f"Results will be saved every {checkpoint_interval} queries")
        
        output_dir = st.text_input("Output Directory", value="./batch_results")
        
        st.markdown("---")
        st.markdown("### 🔧 Processing Options")
        
        enable_validation_retry = st.checkbox("✅ Enable Validation Retry (Step 8)", value=True)
        enable_execution = st.checkbox("⚡ Enable SQL Execution (Step 10)", value=True)
        enable_evaluation = st.checkbox("📊 Enable Evaluation (Step 11)", value=True)
        enable_full_retry = st.checkbox("🔄 Enable Full Pipeline Retry", value=True)
        
        if enable_full_retry:
            st.markdown("**Retry Settings:**")
            max_full_retries = st.slider("Max Full Retries", 0, 5, 2)
            min_eval_score = st.slider("Min Evaluation Score", 0.0, 1.0, 0.8, 0.05)
            st.caption(f"Will retry if EX=0 or score < {min_eval_score}")
        
        st.markdown("---")
        
        # Checkpoint management
        st.markdown("### 💾 Checkpoint Management")
        
        checkpoint_dir = Path(output_dir)
        checkpoint_files = get_checkpoint_files(checkpoint_dir)
        
        if checkpoint_files:
            st.info(f"Found {len(checkpoint_files)} checkpoint files")
            
            if st.button("🔄 Resume from Latest Checkpoint"):
                latest = checkpoint_files[-1]
                checkpoint_data = load_checkpoint(latest)
                if checkpoint_data:
                    st.session_state.batch_results = checkpoint_data['results']
                    st.session_state.batch_timestamp = checkpoint_data['timestamp']
                    st.session_state.checkpoint_resumed = True
                    st.success(f"✅ Resumed from checkpoint with {len(checkpoint_data['results'])} results")
        
        st.markdown("---")
        
        if st.button("🔥 Load Dataset", use_container_width=True):
            data = load_spider_data(spider_json_path)
            if data:
                st.session_state.spider_data = data
                st.success(f"✅ Loaded {len(data)} examples")
        
        if 'spider_data' in st.session_state and st.session_state.spider_data:
            st.info(f"📊 {len(st.session_state.spider_data)} examples loaded")
    
    # Main content
    if 'spider_data' not in st.session_state or not st.session_state.spider_data:
        st.info("👈 Load dataset from sidebar to begin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 How to use:
            1. Configure paths in sidebar
            2. Click "Load Dataset"
            3. Set number of queries and start index
            4. Set checkpoint interval (default: 25)
            5. Choose output directory
            6. Enable processing options
            7. Click "Run Batch Processing"
            
            ### 💾 Automatic Checkpoints:
            - Results saved every N queries
            - Resume from last checkpoint if interrupted
            - No data loss on failures
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Processing Options:
            - **Validation Retry**: Fix errors (Step 8)
            - **Execution**: Run SQL on database (Step 10)
            - **Evaluation**: Compare with ground truth (Step 11)
            - **Full Retry**: Retry if EX=0 or score low
            
            ### 📊 View Features:
            - Summary cards for all queries
            - Full detailed view with all tabs
            - Spider benchmark metrics (EX, EM)
            - Retry history with improvements
            - Export to CSV/JSON
            """)
        
        return
    
    # Batch processing section
    st.markdown("## 🎯 Batch Processing Configuration")
    
    end_idx = min(start_idx + num_queries, len(st.session_state.spider_data))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Start Index", start_idx)
    with col2:
        st.metric("End Index", end_idx - 1)
    with col3:
        st.metric("Total Queries", end_idx - start_idx)
    with col4:
        st.metric("Checkpoint Every", checkpoint_interval)
    
    # Display processing options summary
    st.markdown("**Enabled Options:**")
    options = []
    if enable_validation_retry:
        options.append("✅ Validation Retry")
    if enable_execution:
        options.append("⚡ Execution")
    if enable_evaluation:
        options.append("📊 Evaluation")
    if enable_full_retry:
        options.append(f"🔄 Full Retry (max {max_full_retries}, min score {min_eval_score})")
    
    if options:
        for opt in options:
            st.caption(opt)
    
    st.markdown("---")
    
    if st.button("🚀 Run Batch Processing", type="primary", use_container_width=True):
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ADAPT
        with st.spinner("Initializing ADAPT-SQL pipeline..."):
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
        
        # Results container
        results_container = st.container()
        
        # Initialize or resume results
        if 'batch_results' not in st.session_state or not st.session_state.get('checkpoint_resumed', False):
            results = []
            st.session_state.batch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            results = st.session_state.batch_results
            st.info(f"▶️ Resuming from checkpoint with {len(results)} existing results")
            st.session_state.checkpoint_resumed = False
        
        # Process each query
        for i in range(start_idx, end_idx):
            # Skip if already processed
            if any(r['index'] == i for r in results):
                continue
            
            example = st.session_state.spider_data[i]
            
            # Update status
            progress = (i - start_idx + 1) / (end_idx - start_idx)
            progress_bar.progress(progress)
            status_text.markdown(f"**Processing {i - start_idx + 1}/{end_idx - start_idx}:** {example['question'][:80]}...")
            
            # Get database path
            db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
            
            if not db_path.exists():
                st.warning(f"⚠️ Database not found: {example['db_id']}")
                continue
            
            # Load schema
            try:
                schema_dict = get_schema_from_sqlite(str(db_path))
                foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
            except Exception as e:
                st.error(f"❌ Error loading schema for {example['db_id']}: {e}")
                continue
            
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
                    
                    # Store full retry result
                    results.append({
                        'index': i,
                        'example': example,
                        'result': result,
                        'retry_result': retry_result
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
                st.error(f"❌ Error processing query {i}: {str(e)}")
                import traceback
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())
            
            # Save checkpoint every N queries
            if len(results) % checkpoint_interval == 0:
                checkpoint_path = save_checkpoint(
                    results, 
                    st.session_state.batch_timestamp,
                    output_path
                )
                status_text.markdown(f"💾 **Checkpoint saved:** {checkpoint_path.name} ({len(results)} results)")
        
        # Save final checkpoint
        final_checkpoint = save_checkpoint(
            results,
            st.session_state.batch_timestamp,
            output_path,
            final=True
        )
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.batch_results = results
        
        st.success(f"✅ Batch processing complete! Processed {len(results)} queries.")
        st.info(f"💾 Final results saved to: {final_checkpoint}")
    
    # Display results if available
    if 'batch_results' in st.session_state:
        st.markdown("---")
        st.markdown(f"## 📊 Results Summary")
        st.caption(f"Generated at {st.session_state.batch_timestamp}")
        
        results = st.session_state.batch_results
        
        # Create tabs - UPDATED: Removed Analysis tab, added Query Summary Cards tab
        summary_tab, cards_tab, details_tab, export_tab = st.tabs([
            "📋 Summary View", 
            "📇 Query Summary Cards",
            "🔍 Detailed View", 
            "💾 Export"
        ])
        
        # =====================================================================
        # SUMMARY VIEW TAB - ALL STATISTICS HERE
        # =====================================================================
        with summary_tab:
            # Overall summary metrics
            display_batch_summary(results)
            
            st.markdown("---")
            
            # Complexity distribution
            st.markdown("### 📈 Complexity Distribution")
            display_complexity_distribution(results)
            
            st.markdown("---")
            
            # Execution summary (if enabled)
            if any(r['result'].get('step10_generated') for r in results):
                st.markdown("### ⚡ Execution Statistics")
                display_execution_summary(results)
                st.markdown("---")
            
            # Evaluation summary (if enabled)
            if any(r['result'].get('step11') for r in results):
                st.markdown("### 🎯 Evaluation Statistics (Spider Metrics)")
                display_evaluation_summary(results)
                st.markdown("---")
            
            # Retry summary (if enabled)
            if any(r.get('retry_result') for r in results):
                st.markdown("### 🔄 Retry Statistics")
                display_retry_summary(results)
                st.markdown("---")
            
            # Error analysis
            st.markdown("### 🔍 Error Analysis")
            display_error_analysis(results)
            
            st.markdown("---")
            
            # Performance breakdown
            st.markdown("### 📊 Performance Breakdown")
            display_performance_breakdown(results)
            
            st.markdown("---")
            
            # Score distribution
            if any(r['result'].get('step11') for r in results):
                st.markdown("### 📉 Score Distribution")
                display_score_distribution(results)
        
        # =====================================================================
        # QUERY SUMMARY CARDS TAB - ALL CARDS HERE
        # =====================================================================
        with cards_tab:
            st.markdown("### 📇 Query Summary Cards")
            st.caption("Quick overview of all processed queries")
            
            # Optional: Add filter for cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_valid_only = st.checkbox("Show Valid Only", value=False)
            with col2:
                show_ex_success = st.checkbox("Show EX=1.0 Only", value=False)
            with col3:
                show_complexity = st.multiselect(
                    "Filter Complexity",
                    ["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"],
                    default=["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"]
                )
            
            st.markdown("---")
            
            # Display cards with optional filtering
            cards_displayed = 0
            for r in results:
                # Apply filters
                if show_valid_only and not r['result'].get('final_is_valid', False):
                    continue
                
                if show_ex_success:
                    ex_acc = r['result'].get('step11', {}).get('execution_accuracy', False)
                    if not ex_acc:
                        continue
                
                complexity = r['result']['step2']['complexity_class'].value
                if complexity not in show_complexity:
                    continue
                
                # Display card
                display_query_summary_card(
                    r['index'], 
                    r['example'], 
                    r['result'],
                    r.get('retry_result')
                )
                st.markdown("---")
                cards_displayed += 1
            
            st.caption(f"Showing {cards_displayed} of {len(results)} queries")
        
        # =====================================================================
        # DETAILED VIEW TAB - UNCHANGED
        # =====================================================================
        with details_tab:
            st.markdown("### 🔍 Detailed Query Results with Full UI")
            st.caption("Each query shows all tabs: Schema, Complexity, Examples, Route, SQL, Validation, Execution, Evaluation, Retry History")
            
            # Filter options
            st.markdown("#### 🔎 Filter Options")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                filter_complexity = st.multiselect(
                    "Complexity",
                    ["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"],
                    default=["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"]
                )
            
            with col2:
                filter_validity = st.selectbox(
                    "Validity",
                    ["All", "Valid Only", "Invalid Only"],
                    index=0
                )
            
            with col3:
                filter_execution = st.selectbox(
                    "Execution",
                    ["All", "Success Only", "Failed Only"],
                    index=0
                )
            
            with col4:
                filter_evaluation = st.selectbox(
                    "Evaluation",
                    ["All", "EX = 1.0 Only", "EX = 0.0 Only", "High Score (>=0.7)", "Low Score (<0.5)"],
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
            
            st.markdown("---")
            
            # Display detailed results
            for r in filtered_results:
                display_query_details(
                    r['index'], 
                    r['example'], 
                    r['result'],
                    r.get('retry_result')
                )
        
        # =====================================================================
        # EXPORT TAB - UNCHANGED
        # =====================================================================
        with export_tab:
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 CSV Export")
                st.caption("Summary data suitable for spreadsheets and analysis")
                
                if st.button("🔥 Generate CSV", use_container_width=True):
                    csv = export_summary_csv(results)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"adapt_sql_batch_{st.session_state.batch_timestamp.replace(' ', '_').replace(':', '-')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                st.markdown("#### 📋 JSON Export")
                st.caption("Complete data including all steps and reasoning")
                
                if st.button("🔥 Generate JSON", use_container_width=True):
                    json_str = export_full_json(results)
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_str,
                        file_name=f"adapt_sql_batch_{st.session_state.batch_timestamp.replace(' ', '_').replace(':', '-')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            st.markdown("#### 📊 Export Contents")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**CSV includes:**")
                st.markdown("""
                - Question and database
                - Complexity and strategy
                - Generated and ground truth SQL
                - Validation scores and errors
                - Execution success and time
                - Evaluation scores (EX, EM)
                - Retry attempts and success
                """)
            
            with col2:
                st.markdown("**JSON includes:**")
                st.markdown("""
                - All CSV data PLUS:
                - Complete reasoning for each step
                - Schema linking details
                - Similar examples used
                - Full validation feedback
                - Retry history with improvements
                - All intermediate representations
                """)


if __name__ == "__main__":
    main()