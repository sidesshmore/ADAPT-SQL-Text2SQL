"""
ADAPT-SQL Streamlit Application - With Step 8 Validation Retry & Batch Processing
"""
import streamlit as st
import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from adapt_baseline import ADAPTBaseline


st.set_page_config(page_title="ADAPT-SQL Pipeline", page_icon="🎯", layout="wide")

if 'spider_data' not in st.session_state:
    st.session_state.spider_data = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None


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


def display_validation_badge(is_valid, validation_score):
    """Display validation status badge"""
    if is_valid:
        st.success(f"✅ Valid SQL (Score: {validation_score:.2f})")
    else:
        st.error(f"❌ Invalid SQL (Score: {validation_score:.2f})")


def run_batch_processing(
    spider_data, 
    spider_db_dir, 
    model, 
    vector_store_path, 
    k_examples, 
    num_queries,
    enable_retry,
    max_retries
):
    """Run batch processing on multiple queries"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(min(num_queries, len(spider_data))):
        example = spider_data[i]
        
        status_text.text(f"Processing {i+1}/{num_queries}: {example['question'][:50]}...")
        
        try:
            # Get schema and foreign keys
            db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
            
            if not db_path.exists():
                results.append({
                    'question_id': i,
                    'question': example['question'],
                    'db_id': example['db_id'],
                    'status': 'ERROR',
                    'error': 'Database file not found',
                    'ground_truth': example.get('query', 'N/A')
                })
                continue
            
            schema_dict = get_schema_from_sqlite(str(db_path))
            foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
            
            # Initialize ADAPT
            adapt = ADAPTBaseline(
                model=model, 
                vector_store_path=vector_store_path,
                max_retries=max_retries
            )
            
            # Run pipeline
            result = adapt.run_full_pipeline(
                example['question'],
                schema_dict,
                foreign_keys,
                k_examples=k_examples,
                enable_retry=enable_retry
            )
            
            # Extract results
            results.append({
                'question_id': i,
                'question': example['question'],
                'db_id': example['db_id'],
                'complexity': result['step2']['complexity_class'].value,
                'strategy': result['step5']['strategy'].value,
                'generated_sql': result['final_sql'],
                'is_valid': result['final_is_valid'],
                'validation_score': result['step7']['validation_score'],
                'retry_count': result['step8']['retry_count'] if result.get('step8') else 0,
                'ground_truth': example.get('query', 'N/A'),
                'status': 'SUCCESS'
            })
            
        except Exception as e:
            results.append({
                'question_id': i,
                'question': example['question'],
                'db_id': example['db_id'],
                'status': 'ERROR',
                'error': str(e),
                'ground_truth': example.get('query', 'N/A')
            })
        
        progress_bar.progress((i + 1) / num_queries)
    
    status_text.text("✅ Batch processing completed!")
    
    return results


def display_batch_results(results):
    """Display batch processing results"""
    st.markdown("### 📊 Batch Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(results)
    successful = len([r for r in results if r['status'] == 'SUCCESS'])
    valid = len([r for r in results if r.get('is_valid', False)])
    avg_score = sum([r.get('validation_score', 0) for r in results if r['status'] == 'SUCCESS']) / max(successful, 1)
    avg_retries = sum([r.get('retry_count', 0) for r in results if r['status'] == 'SUCCESS']) / max(successful, 1)
    
    with col1:
        st.metric("Total Queries", total)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Valid SQL", valid)
    with col4:
        st.metric("Avg Score", f"{avg_score:.2f}")
    with col5:
        st.metric("Avg Retries", f"{avg_retries:.2f}")
    
    # Complexity breakdown
    st.markdown("#### Complexity Distribution")
    complexity_counts = {}
    for r in results:
        if r['status'] == 'SUCCESS':
            comp = r.get('complexity', 'UNKNOWN')
            complexity_counts[comp] = complexity_counts.get(comp, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 EASY", complexity_counts.get('EASY', 0))
    with col2:
        st.metric("🟡 NON_NESTED_COMPLEX", complexity_counts.get('NON_NESTED_COMPLEX', 0))
    with col3:
        st.metric("🔴 NESTED_COMPLEX", complexity_counts.get('NESTED_COMPLEX', 0))
    
    # Results table
    st.markdown("#### Detailed Results")
    
    df_data = []
    for r in results:
        df_data.append({
            'ID': r['question_id'],
            'Question': r['question'][:50] + '...',
            'DB': r['db_id'],
            'Status': r['status'],
            'Complexity': r.get('complexity', 'N/A'),
            'Valid': '✅' if r.get('is_valid', False) else '❌',
            'Score': f"{r.get('validation_score', 0):.2f}",
            'Retries': r.get('retry_count', 0)
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download results
    st.markdown("#### 💾 Export Results")
    
    # JSON export
    json_str = json.dumps(results, indent=2)
    st.download_button(
        label="📥 Download Results (JSON)",
        data=json_str,
        file_name=f"adapt_sql_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    # CSV export
    csv_str = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv_str,
        file_name=f"adapt_sql_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def main():
    st.title("🎯 ADAPT-SQL Pipeline")
    st.markdown("Complete Text-to-SQL with Steps 1-8 (including Validation & Retry)")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        mode = st.radio("Mode", ["Single Query", "Batch Processing"])
        
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
        
        # Step 8 Configuration
        st.markdown("---")
        st.markdown("**Step 8: Validation Retry**")
        enable_retry = st.checkbox("Enable Retry", value=True)
        max_retries = st.slider("Max Retries", 1, 5, 2)
        
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
    
    # SINGLE QUERY MODE
    if mode == "Single Query":
        st.header("🔍 Query Analysis")
        
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
                
                adapt = ADAPTBaseline(
                    model=model, 
                    vector_store_path=vector_store_path,
                    max_retries=max_retries
                )
                
                result = adapt.run_full_pipeline(
                    example['question'],
                    schema_dict,
                    foreign_keys,
                    k_examples=k_examples,
                    enable_retry=enable_retry
                )
                
                st.success("✅ Complete!")
                st.markdown("---")
                
                # Display results - NOW WITH 7 TABS INCLUDING RETRY
                tabs = st.tabs([
                    "📊 Schema", "🔍 Complexity", "🔎 Examples", 
                    "🔀 Route", "✨ SQL", "✅ Validation", "🔄 Retry"
                ])
                
                # [Previous tab implementations remain the same until Validation tab]
                # ... [tabs 1-5 code from original app.py] ...
                
                with tabs[0]:
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
                
                with tabs[1]:
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
                    
                    if result['step2'].get('sub_questions'):
                        st.markdown("**Sub-questions:**")
                        for i, sq in enumerate(result['step2']['sub_questions'], 1):
                            st.info(f"{i}. {sq}")
                    
                    st.markdown("**Preliminary SQL:**")
                    st.code(result['step3']['predicted_sql'], language='sql')
                
                with tabs[2]:
                    st.markdown("### Similar Examples")
                    st.metric("Found", result['step4']['total_found'])
                    
                    for i, ex in enumerate(result['step4']['similar_examples'][:5], 1):
                        score = ex.get('similarity_score', 0)
                        color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                        
                        with st.expander(f"{color} {i}. {ex.get('question', '')[:60]}... ({score:.3f})"):
                            st.markdown(f"**Question:** {ex.get('question', '')}")
                            st.code(ex.get('query', ''), language='sql')
                
                with tabs[3]:
                    st.markdown("### Routing Strategy")
                    strategy = result['step5']['strategy'].value
                    st.success(f"🎯 {strategy}")
                    st.info(result['step5']['description'])
                
                with tabs[4]:
                    st.markdown("### Generated SQL")
                    
                    # Display generated SQL based on strategy
                    if result.get('step6a'):
                        st.markdown("**Method:** Simple Few-Shot (6a)")
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
                    
                    elif result.get('step6b'):
                        st.markdown("**Method:** Intermediate Representation (6b)")
                        
                        with st.expander("🔍 NatSQL Intermediate"):
                            st.code(result['step6b']['natsql_intermediate'], language='text')
                        
                        st.markdown("**Generated SQL:**")
                        st.code(result['step6b']['generated_sql'], language='sql')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            conf = result['step6b']['confidence']
                            if conf >= 0.8:
                                st.success(f"Confidence: {conf:.1%}")
                            elif conf >= 0.6:
                                st.warning(f"Confidence: {conf:.1%}")
                            else:
                                st.error(f"Confidence: {conf:.1%}")
                        with col2:
                            st.metric("Examples Used", result['step6b']['examples_used'])
                    
                    elif result.get('step6c'):
                        st.markdown("**Method:** Decomposed Generation (6c)")
                        
                        if result['step6c']['sub_sql_list']:
                            st.markdown("**Sub-queries:**")
                            for i, sub_info in enumerate(result['step6c']['sub_sql_list'], 1):
                                with st.expander(f"Sub-query {i}: {sub_info['sub_question'][:40]}... [{sub_info['complexity']}]"):
                                    st.markdown(f"**Question:** {sub_info['sub_question']}")
                                    st.code(sub_info['sql'], language='sql')
                        
                        with st.expander("🔍 NatSQL Intermediate with Sub-queries"):
                            st.code(result['step6c']['natsql_intermediate'], language='text')
                        
                        st.markdown("**Final SQL:**")
                        st.code(result['step6c']['generated_sql'], language='sql')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            conf = result['step6c']['confidence']
                            if conf >= 0.75:
                                st.success(f"Confidence: {conf:.1%}")
                            elif conf >= 0.55:
                                st.warning(f"Confidence: {conf:.1%}")
                            else:
                                st.error(f"Confidence: {conf:.1%}")
                        with col2:
                            st.metric("Examples Used", result['step6c']['examples_used'])
                        with col3:
                            st.metric("Sub-queries", len(result['step6c']['sub_sql_list']))
                    
                    else:
                        st.warning("⚠️ No SQL generated")
                    
                    # Compare with ground truth
                    if 'query' in example:
                        st.markdown("---")
                        st.markdown("**Compare with Ground Truth:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("*Generated:*")
                            generated = (result.get('step6a') or result.get('step6b') or result.get('step6c') or {}).get('generated_sql', 'N/A')
                            st.code(generated, language='sql')
                        with col2:
                            st.markdown("*Ground Truth:*")
                            st.code(example['query'], language='sql')
                
                # Validation Tab (Step 7)
                with tabs[5]:
                    st.markdown("### SQL Validation (Step 7)")
                    
                    if result.get('step7'):
                        validation = result['step7']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            display_validation_badge(validation['is_valid'], validation['validation_score'])
                        with col2:
                            st.metric("Errors", len(validation['errors']))
                        with col3:
                            st.metric("Warnings", len(validation['warnings']))
                        
                        st.markdown("---")
                        
                        if validation['errors']:
                            st.markdown("### ❌ Errors")
                            for i, error in enumerate(validation['errors'], 1):
                                severity_color = {
                                    'CRITICAL': '🔴',
                                    'HIGH': '🟠',
                                    'MEDIUM': '🟡',
                                    'LOW': '🟢'
                                }.get(error['severity'], '⚪')
                                
                                with st.expander(f"{severity_color} Error {i}: {error['type']} [{error['severity']}]", expanded=True):
                                    st.error(error['message'])
                                    
                                    if 'table' in error:
                                        st.write(f"**Table:** `{error['table']}`")
                                    if 'column' in error:
                                        st.write(f"**Column:** `{error['column']}`")
                        else:
                            st.success("✅ No errors found!")
                        
                        st.markdown("---")
                        
                        if validation['warnings']:
                            st.markdown("### ⚠️ Warnings")
                            for i, warning in enumerate(validation['warnings'], 1):
                                severity_color = {
                                    'MEDIUM': '🟡',
                                    'LOW': '🟢'
                                }.get(warning['severity'], '⚪')
                                
                                with st.expander(f"{severity_color} Warning {i}: {warning['type']} [{warning['severity']}]"):
                                    st.warning(warning['message'])
                                    
                                    if 'table' in warning:
                                        st.write(f"**Table:** `{warning['table']}`")
                        else:
                            st.info("No warnings")
                        
                        st.markdown("---")
                        
                        if validation['suggestions']:
                            st.markdown("### 💡 Suggestions")
                            for i, suggestion in enumerate(validation['suggestions'], 1):
                                st.info(f"{i}. {suggestion}")
                        
                        with st.expander("📋 Full Validation Report"):
                            st.text(validation['reasoning'])
                    
                    else:
                        st.warning("⚠️ Validation not performed")
                
                # NEW TAB: Retry (Step 8)
                with tabs[6]:
                    st.markdown("### Validation-Feedback Retry (Step 8)")
                    
                    if result.get('step8'):
                        retry = result['step8']
                        
                        # Overall retry status
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            display_validation_badge(retry['is_valid'], 
                                                   retry['validation_history'][-1]['validation_score'])
                        with col2:
                            st.metric("Retry Attempts", retry['retry_count'])
                        with col3:
                            initial_errors = len(retry['validation_history'][0]['errors'])
                            final_errors = len(retry['validation_history'][-1]['errors'])
                            st.metric("Errors Fixed", initial_errors - final_errors)
                        with col4:
                            if retry['retry_count'] > 0:
                                initial_score = retry['validation_history'][0]['validation_score']
                                final_score = retry['validation_history'][-1]['validation_score']
                                improvement = final_score - initial_score
                                st.metric("Score Improvement", f"+{improvement:.2f}" if improvement > 0 else f"{improvement:.2f}")
                        
                        st.markdown("---")
                        
                        # Validation History
                        if retry['retry_count'] > 0:
                            st.markdown("### 📈 Validation History")
                            
                            for i, validation in enumerate(retry['validation_history']):
                                attempt = "Initial" if i == 0 else f"Retry {i}"
                                
                                with st.expander(f"{attempt} - Score: {validation['validation_score']:.2f} - {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Errors", len(validation['errors']))
                                        st.metric("Warnings", len(validation['warnings']))
                                    with col2:
                                        st.metric("Validation Score", f"{validation['validation_score']:.2f}")
                                        if i > 0:
                                            st.info(retry['improvements'][i-1])
                                    
                                    # Show errors for this attempt
                                    if validation['errors']:
                                        st.markdown("**Errors:**")
                                        for error in validation['errors'][:3]:
                                            st.text(f"• {error['type']}: {error['message']}")
                        else:
                            st.info("✅ No retry needed - SQL was valid on first attempt")
                        
                        st.markdown("---")
                        
                        # Final SQL
                        st.markdown("### 🎯 Final SQL")
                        st.code(retry['final_sql'], language='sql')
                        
                        # Compare initial vs final
                        if retry['retry_count'] > 0:
                            st.markdown("---")
                            st.markdown("### 📊 Initial vs Final Comparison")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Initial SQL:**")
                                initial_sql = (result.get('step6a') or result.get('step6b') or result.get('step6c') or {}).get('generated_sql', 'N/A')
                                st.code(initial_sql, language='sql')
                            with col2:
                                st.markdown("**Final SQL (After Retry):**")
                                st.code(retry['final_sql'], language='sql')
                        
                        # Full reasoning
                        with st.expander("📋 Full Retry Report"):
                            st.text(retry['reasoning'])
                    
                    else:
                        if enable_retry:
                            st.warning("⚠️ Retry was not performed (validation passed or retry disabled)")
                        else:
                            st.info("ℹ️  Retry is disabled in configuration")
    
    # BATCH PROCESSING MODE
    else:
        st.header("📦 Batch Processing")
        
        num_queries = st.number_input(
            "Number of queries to process",
            min_value=1,
            max_value=len(st.session_state.spider_data),
            value=min(10, len(st.session_state.spider_data))
        )
        
        st.info(f"Will process queries 1-{num_queries} from the dataset")
        
        if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
            results = run_batch_processing(
                st.session_state.spider_data,
                spider_db_dir,
                model,
                vector_store_path,
                k_examples,
                num_queries,
                enable_retry,
                max_retries
            )
            
            st.session_state.batch_results = results
        
        # Display results if available
        if st.session_state.batch_results:
            st.markdown("---")
            display_batch_results(st.session_state.batch_results)


if __name__ == "__main__":
    main()