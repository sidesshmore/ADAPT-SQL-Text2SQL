"""
ADAPT-SQL Streamlit Application - Streamlined Version
"""
import streamlit as st
import json
import sqlite3
import pandas as pd
from pathlib import Path
from adapt_baseline import ADAPTBaseline
from datetime import datetime


st.set_page_config(
    page_title="ADAPT-SQL Pipeline",
    page_icon="🎯",
    layout="wide"
)

# Session state
if 'spider_data' not in st.session_state:
    st.session_state.spider_data = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []


def load_spider_data(json_path: str):
    """Load Spider dataset"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error: {e}")
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
        st.error(f"Error: {e}")
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
        st.error(f"Error: {e}")
        return []


def display_complexity(complexity):
    """Display complexity with color"""
    if complexity == "EASY":
        st.success(f"🟢 {complexity}")
    elif complexity == "NON_NESTED_COMPLEX":
        st.warning(f"🟡 {complexity}")
    else:
        st.error(f"🔴 {complexity}")


def process_single_example(adapt, example, spider_db_dir, k_examples):
    """Process a single example"""
    db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
    
    if not db_path.exists():
        return {'status': 'error', 'error': f"Database not found"}
    
    try:
        start_time = datetime.now()
        
        schema_dict = get_schema_from_sqlite(str(db_path))
        foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
        
        result = adapt.run_full_pipeline(
            example['question'],
            schema_dict,
            foreign_keys,
            k_examples=k_examples
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'status': 'success',
            'result': result,
            'time': processing_time,
            'complexity': result['step2']['complexity_class'].value
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def main():
    st.title("🎯 ADAPT-SQL Pipeline")
    st.markdown("End-to-end Text-to-SQL Generation")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model = st.selectbox("Model", ["llama3.2", "codellama", "mistral", "qwen2.5"])
        
        spider_json_path = st.text_input(
            "Spider dev.json",
            value="/Users/sidessh/ADAPT-SQL/data/spider/dev.json"
        )
        
        spider_db_dir = st.text_input(
            "Spider DB directory",
            value="/Users/sidessh/ADAPT-SQL/data/spider/spider_data/database"
        )
        
        vector_store_path = st.text_input(
            "Vector Store",
            value="./vector_store"
        )
        
        k_examples = st.slider("Similar Examples", 1, 20, 10)
        
        st.markdown("---")
        
        if st.button("📂 Load Dataset"):
            data = load_spider_data(spider_json_path)
            if data:
                st.session_state.spider_data = data
                st.success(f"✅ {len(data)} examples loaded")
        
        if st.session_state.spider_data:
            st.info(f"📊 {len(st.session_state.spider_data)} examples")
    
    if not st.session_state.spider_data:
        st.info("👈 Load dataset from sidebar")
        return
    
    # Mode selection
    mode = st.radio("Mode", ["Single Query", "Batch Processing", "View Results"], horizontal=True)
    
    if mode == "Single Query":
        st.header("🔍 Single Query")
        
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
        
        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
                
                if not db_path.exists():
                    st.error("❌ Database not found")
                    return
                
                schema_dict = get_schema_from_sqlite(str(db_path))
                foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
                
                adapt = ADAPTBaseline(model=model, vector_store_path=vector_store_path)
                
                result = adapt.run_full_pipeline(
                    example['question'],
                    schema_dict,
                    foreign_keys,
                    k_examples=k_examples
                )
                
                st.success("✅ Complete!")
                st.markdown("---")
                
                # Display results
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Schema", "🔍 Complexity", "🔎 Examples", "🔀 Route", "✨ SQL"
                ])
                
                with tab1:
                    st.markdown("### Schema Linking")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tables", len(result['step1']['schema_links']['tables']))
                    with col2:
                        total_cols = sum(len(cols) for cols in result['step1']['schema_links']['columns'].values())
                        st.metric("Columns", total_cols)
                    with col3:
                        st.metric("Foreign Keys", len(result['step1']['schema_links']['foreign_keys']))
                    
                    st.markdown("**Tables:**")
                    for table in sorted(result['step1']['schema_links']['tables']):
                        st.success(f"📊 {table}")
                
                with tab2:
                    st.markdown("### Complexity Classification")
                    display_complexity(result['step2']['complexity_class'].value)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"• Tables: {len(result['step2']['required_tables'])}")
                        st.write(f"• JOINs: {'✅' if result['step2']['needs_joins'] else '❌'}")
                        st.write(f"• Subqueries: {'✅' if result['step2']['needs_subqueries'] else '❌'}")
                    with col2:
                        if result['step2']['aggregations']:
                            st.write(f"• Aggregations: {', '.join(result['step2']['aggregations'])}")
                        st.write(f"• GROUP BY: {'✅' if result['step2']['has_grouping'] else '❌'}")
                    
                    st.markdown("**Preliminary SQL:**")
                    st.code(result['step3']['predicted_sql'], language='sql')
                
                with tab3:
                    st.markdown("### Similar Examples")
                    st.metric("Found", result['step4']['total_found'])
                    
                    for i, ex in enumerate(result['step4']['similar_examples'][:5], 1):
                        score = ex.get('similarity_score', 0)
                        color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                        
                        with st.expander(f"{color} {i}. {ex.get('question', '')[:60]}... ({score:.3f})"):
                            st.markdown(f"**Question:** {ex.get('question', '')}")
                            st.code(ex.get('query', ''), language='sql')
                
                with tab4:
                    st.markdown("### Routing Strategy")
                    strategy = result['step5']['strategy'].value
                    st.success(f"🎯 {strategy}")
                    st.info(result['step5']['description'])
                
                with tab5:
                    st.markdown("### Generated SQL")
                    if result.get('step6a'):
                        st.code(result['step6a']['generated_sql'], language='sql')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            conf = result['step6a']['confidence']
                            if conf >= 0.8:
                                st.success(f"Confidence: {conf:.1%}")
                            elif conf >= 0.6:
                                st.warning(f"Confidence: {conf:.1%}")
                            else:
                                st.error(f"Confidence: {conf:.1%}")
                        with col2:
                            st.metric("Examples Used", result['step6a']['examples_used'])
                    else:
                        st.warning(f"⚠️ {result['step5']['strategy'].value} not implemented yet")
    
    elif mode == "Batch Processing":
        st.header("📦 Batch Processing")
        
        col1, col2 = st.columns(2)
        with col1:
            start_idx = st.number_input(
                "Start Index",
                min_value=0,
                max_value=len(st.session_state.spider_data) - 1,
                value=0
            )
        with col2:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=100,
                value=10
            )
        
        end_idx = min(start_idx + batch_size, len(st.session_state.spider_data))
        st.info(f"Processing examples {start_idx} to {end_idx - 1}")
        
        if st.button("🚀 Start Batch", type="primary", use_container_width=True):
            st.session_state.batch_results = []
            
            adapt = ADAPTBaseline(model=model, vector_store_path=vector_store_path)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx in range(start_idx, end_idx):
                example = st.session_state.spider_data[idx]
                status_text.text(f"Processing {idx - start_idx + 1}/{end_idx - start_idx}: {example['question'][:50]}...")
                
                process_result = process_single_example(adapt, example, spider_db_dir, k_examples)
                
                result_entry = {
                    'index': idx,
                    'question': example['question'],
                    'db_id': example.get('db_id', ''),
                    'status': process_result['status'],
                    'time': process_result.get('time', 0),
                    'complexity': process_result.get('complexity', 'N/A'),
                    'error': process_result.get('error', None)
                }
                
                if process_result['status'] == 'success':
                    result_entry['generated_sql'] = process_result['result'].get('step6a', {}).get('generated_sql', 'N/A')
                    result_entry['ground_truth_sql'] = example.get('query', 'N/A')
                
                st.session_state.batch_results.append(result_entry)
                progress_bar.progress((idx - start_idx + 1) / (end_idx - start_idx))
            
            status_text.text("✅ Complete!")
            st.success(f"Processed {end_idx - start_idx} examples")
    
    else:
        st.header("📊 Batch Results")
        
        if not st.session_state.batch_results:
            st.info("No results yet")
            return
        
        results = st.session_state.batch_results
        
        # Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", len(results))
        with col2:
            successful = sum(1 for r in results if r.get('status') == 'success')
            st.metric("Success", successful)
        with col3:
            failed = sum(1 for r in results if r.get('status') == 'error')
            st.metric("Failed", failed)
        with col4:
            avg_time = sum(r.get('time', 0) for r in results) / len(results)
            st.metric("Avg Time", f"{avg_time:.1f}s")
        
        st.markdown("---")
        
        # Table
        df_data = []
        for r in results:
            df_data.append({
                'Index': r.get('index'),
                'Question': r.get('question', '')[:50] + '...',
                'Database': r.get('db_id', ''),
                'Complexity': r.get('complexity', ''),
                'Status': r.get('status', ''),
                'Time': f"{r.get('time', 0):.1f}s"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Download
        st.markdown("---")
        results_json = json.dumps(results, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=results_json,
            file_name=f"adapt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()