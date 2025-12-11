"""
ADAPT-SQL Streamlined Batch Processing
Focus on comprehensive at-a-glance statistics
Auto-generates CSV on completion
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
    display_comprehensive_statistics,
    generate_comprehensive_csv,
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
    st.title("📦 ADAPT-SQL Batch Processing")
    st.markdown("Process multiple queries with automatic checkpoints and comprehensive statistics")
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
            "📚 Vector Store",
            value="./vector_store"
        )
        
        k_examples = st.slider("📖 Similar Examples", 1, 20, 10)
        
        st.markdown("---")
        st.markdown("### 🎯 Batch Settings")
        
        num_queries = st.number_input("Number of Queries", min_value=1, max_value=1000, value=10, step=5)
        start_idx = st.number_input("Start Index", min_value=0, value=0)
        
        checkpoint_interval = st.number_input("Checkpoint Interval", min_value=5, max_value=100, value=25, step=5)
        st.caption(f"Saves progress every {checkpoint_interval} queries")
        
        # Fixed directories
        checkpoint_dir = "./batch_results"
        results_csv_dir = "./results"
        
        st.info(f"📂 Checkpoints: `{checkpoint_dir}`")
        st.info(f"📊 CSV Results: `{results_csv_dir}`")
        
        st.markdown("---")
        st.markdown("### 🔧 Processing Options")
        
        enable_validation_retry = st.checkbox("✅ Validation Retry (Step 8)", value=True)
        enable_execution = st.checkbox("⚡ SQL Execution (Step 10)", value=True)
        enable_evaluation = st.checkbox("📊 Evaluation (Step 11)", value=True)
        enable_full_retry = st.checkbox("🔄 Full Pipeline Retry", value=True)
        
        if enable_full_retry:
            st.markdown("**Retry Settings:**")
            max_full_retries = st.slider("Max Full Retries", 0, 5, 2)
            min_eval_score = st.slider("Min Evaluation Score", 0.0, 1.0, 0.8, 0.05)
            st.caption(f"Retries if EX=0 or score < {min_eval_score}")
        
        st.markdown("---")
        
        # Checkpoint management
        st.markdown("### 💾 Checkpoint Management")
        
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_files = get_checkpoint_files(checkpoint_path)
        
        if checkpoint_files:
            st.success(f"✅ Found {len(checkpoint_files)} checkpoint files")
            
            # Show most recent checkpoint info
            latest = checkpoint_files[0]
            checkpoint_data = load_checkpoint(latest)
            if checkpoint_data:
                st.caption(f"Latest: {checkpoint_data['num_results']} queries")
                st.caption(f"Saved: {checkpoint_data['saved_at']}")
            
            if st.button("🔄 Resume from Latest", use_container_width=True):
                checkpoint_data = load_checkpoint(latest)
                if checkpoint_data:
                    st.session_state.batch_results = checkpoint_data['results']
                    st.session_state.batch_timestamp = checkpoint_data['timestamp']
                    st.session_state.checkpoint_resumed = True
                    st.success(f"✅ Resumed: {len(checkpoint_data['results'])} results")
        else:
            st.info("📂 No checkpoints found")
            st.caption("Checkpoints will appear here after running batch processing")
            st.caption(f"Looking in: {checkpoint_path.absolute()}")
        
        st.markdown("---")
        
        if st.button("📥 Load Dataset", use_container_width=True):
            data = load_spider_data(spider_json_path)
            if data:
                st.session_state.spider_data = data
                st.success(f"✅ Loaded {len(data)} examples")
        
        if 'spider_data' in st.session_state and st.session_state.spider_data:
            st.info(f"📊 {len(st.session_state.spider_data)} examples loaded")
    
    # Main content
    if 'spider_data' not in st.session_state or not st.session_state.spider_data:
        st.info("👈 Load dataset from sidebar to begin")
        
        st.markdown("""
        ### 🚀 Quick Start Guide:
        
        1. **Configure Paths** (sidebar)
           - Set Spider dev.json path
           - Set database directory
           - Set vector store path
        
        2. **Load Dataset**
           - Click "Load Dataset" button
        
        3. **Configure Batch**
           - Set number of queries
           - Set checkpoint interval (default: 25)
        
        4. **Enable Options**
           - Validation retry (Step 8)
           - Execution (Step 10) 
           - Evaluation (Step 11)
           - Full pipeline retry
        
        5. **Run Processing**
           - Click "Run Batch Processing"
           - CSV auto-generated on completion
        
        ### 📊 Features:
        
        - **Comprehensive Statistics**: At-a-glance system performance overview
        - **Automatic Checkpoints**: Progress saved to `./batch_results` every N queries
        - **Resume Capability**: Continue from last checkpoint
        - **Auto CSV Export**: Results automatically saved to `./results`
        - **Spider Metrics**: Official EX and EM benchmark scores
        - **Error Analysis**: Detailed breakdown of common issues
        """)
        
        return
    
    # Batch processing section
    st.markdown("## 🎯 Batch Configuration")
    
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
    
    # Display enabled options
    st.markdown("**Enabled:**")
    options_text = []
    if enable_validation_retry:
        options_text.append("✅ Validation Retry")
    if enable_execution:
        options_text.append("⚡ Execution")
    if enable_evaluation:
        options_text.append("📊 Evaluation")
    if enable_full_retry:
        options_text.append(f"🔄 Full Retry (max {max_full_retries}, score ≥{min_eval_score})")
    
    st.caption(" | ".join(options_text))
    
    st.markdown("---")
    
    if st.button("🚀 Run Batch Processing", type="primary", use_container_width=True):
        # Create output directories
        checkpoint_path = Path(checkpoint_dir)
        results_path = Path(results_csv_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)
        
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
        
        # Initialize or resume results
        if 'batch_results' not in st.session_state or not st.session_state.get('checkpoint_resumed', False):
            results = []
            st.session_state.batch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            results = st.session_state.batch_results
            st.info(f"▶️ Resuming from checkpoint: {len(results)} existing results")
            st.session_state.checkpoint_resumed = False
        
        # Process each query
        for i in range(start_idx, end_idx):
            # Skip if already processed
            if any(r['index'] == i for r in results):
                continue
            
            example = st.session_state.spider_data[i]
            
            # Update progress
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
                checkpoint_file = save_checkpoint(
                    results, 
                    st.session_state.batch_timestamp,
                    checkpoint_path
                )
                status_text.markdown(f"💾 **Checkpoint saved:** {len(results)} results")
        
        # Save final checkpoint
        final_checkpoint = save_checkpoint(
            results,
            st.session_state.batch_timestamp,
            checkpoint_path,
            final=True
        )
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.batch_results = results
        
        # Generate CSV automatically
        st.markdown("---")
        st.markdown("### 💾 Generating Results CSV...")
        
        with st.spinner("Creating comprehensive CSV report..."):
            csv_path = generate_comprehensive_csv(
                results,
                st.session_state.batch_timestamp,
                results_path
            )
        
        st.success(f"✅ Batch processing complete! Processed {len(results)} queries.")
        st.success(f"📊 CSV saved to: **{csv_path}**")
        st.info(f"💾 Checkpoint saved to: {final_checkpoint}")
        
        # Offer download button
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_data = f.read()
        
        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv_data,
            file_name=csv_path.name,
            mime="text/csv",
            use_container_width=True
        )
    
    # Display results if available
    if 'batch_results' in st.session_state:
        st.markdown("---")
        
        results = st.session_state.batch_results
        
        # Display comprehensive statistics
        display_comprehensive_statistics(results)


if __name__ == "__main__":
    main()