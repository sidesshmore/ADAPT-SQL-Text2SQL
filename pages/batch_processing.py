"""
ADAPT-SQL Batch Processing Page
Process multiple queries and display detailed results
"""
import streamlit as st
import json
import sqlite3
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path to import adapt_baseline
sys.path.append(str(Path(__file__).parent.parent))
from adapt_baseline import ADAPTBaseline


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


def display_query_details(idx: int, example: dict, result: dict):
    """Display detailed results for a single query"""
    with st.expander(f"🔍 Query #{idx + 1}: {example['question'][:80]}...", expanded=False):
        st.markdown(f"**Database:** `{example['db_id']}`")
        st.markdown(f"**Question:** {example['question']}")
        
        # Create tabs for different aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Summary", "🔗 Schema", "🎯 SQL", "✅ Validation", "📋 Full Details"
        ])
        
        with tab1:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                complexity = result['step2']['complexity_class'].value
                if complexity == "EASY":
                    st.success(f"🟢 {complexity}")
                elif complexity == "NON_NESTED_COMPLEX":
                    st.warning(f"🟡 {complexity}")
                else:
                    st.error(f"🔴 {complexity}")
            
            with col2:
                st.metric("Tables", len(result['step1']['schema_links']['tables']))
            
            with col3:
                strategy = result['step5']['strategy'].value.replace('_', ' ')
                st.info(f"Strategy: {strategy}")
            
            with col4:
                if result.get('step7'):
                    val_score = result['step7']['validation_score']
                    if result['step7']['is_valid']:
                        st.success(f"✅ Valid ({val_score:.2f})")
                    else:
                        st.error(f"❌ Invalid ({val_score:.2f})")
        
        with tab2:
            # Schema linking details
            st.markdown("**Relevant Tables:**")
            for table in sorted(result['step1']['schema_links']['tables']):
                st.success(f"📊 {table}")
            
            if result['step1']['schema_links']['foreign_keys']:
                st.markdown("**Foreign Keys:**")
                for fk in result['step1']['schema_links']['foreign_keys']:
                    st.info(f"{fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}")
        
        with tab3:
            # Generated SQL
            generated_sql = None
            if result.get('step6a'):
                st.markdown("**Method:** Simple Few-Shot (6a)")
                generated_sql = result['step6a']['generated_sql']
            elif result.get('step6b'):
                st.markdown("**Method:** Intermediate Representation (6b)")
                with st.expander("🔍 NatSQL Intermediate"):
                    st.code(result['step6b']['natsql_intermediate'], language='text')
                generated_sql = result['step6b']['generated_sql']
            elif result.get('step6c'):
                st.markdown("**Method:** Decomposed Generation (6c)")
                if result['step6c']['sub_sql_list']:
                    st.markdown(f"**Sub-queries:** {len(result['step6c']['sub_sql_list'])}")
                generated_sql = result['step6c']['generated_sql']
            
            if generated_sql:
                st.code(generated_sql, language='sql')
            
            # Ground truth comparison
            if 'query' in example:
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Generated SQL:**")
                    st.code(generated_sql if generated_sql else "N/A", language='sql')
                with col2:
                    st.markdown("**Ground Truth:**")
                    st.code(example['query'], language='sql')
        
        with tab4:
            # Validation results
            if result.get('step7'):
                validation = result['step7']
                
                # Errors
                if validation['errors']:
                    st.markdown("### ❌ Errors")
                    for i, error in enumerate(validation['errors'], 1):
                        severity_color = {
                            'CRITICAL': '🔴',
                            'HIGH': '🟠',
                            'MEDIUM': '🟡',
                            'LOW': '🟢'
                        }.get(error['severity'], '⚪')
                        
                        st.error(f"{severity_color} **{error['type']}** [{error['severity']}]: {error['message']}")
                else:
                    st.success("✅ No errors found!")
                
                # Warnings
                if validation['warnings']:
                    st.markdown("### ⚠️ Warnings")
                    for i, warning in enumerate(validation['warnings'], 1):
                        st.warning(f"**{warning['type']}**: {warning['message']}")
                
                # Suggestions
                if validation['suggestions']:
                    st.markdown("### 💡 Suggestions")
                    for suggestion in validation['suggestions']:
                        st.info(suggestion)
            else:
                st.warning("⚠️ Validation not performed")
        
        with tab5:
            # Full pipeline details
            st.json({
                'step1_schema_links': {
                    'tables': list(result['step1']['schema_links']['tables']),
                    'num_columns': sum(len(cols) for cols in result['step1']['schema_links']['columns'].values()),
                    'foreign_keys': len(result['step1']['schema_links']['foreign_keys'])
                },
                'step2_complexity': {
                    'class': result['step2']['complexity_class'].value,
                    'needs_joins': result['step2']['needs_joins'],
                    'needs_subqueries': result['step2']['needs_subqueries'],
                    'aggregations': result['step2']['aggregations']
                },
                'step4_examples': {
                    'total_found': result['step4']['total_found'],
                    'top_similarity': result['step4']['similar_examples'][0]['similarity_score'] if result['step4']['similar_examples'] else 0
                },
                'step5_routing': {
                    'strategy': result['step5']['strategy'].value,
                    'description': result['step5']['description']
                },
                'step7_validation': {
                    'is_valid': result['step7']['is_valid'] if result.get('step7') else None,
                    'score': result['step7']['validation_score'] if result.get('step7') else None,
                    'num_errors': len(result['step7']['errors']) if result.get('step7') else None,
                    'num_warnings': len(result['step7']['warnings']) if result.get('step7') else None
                }
            })


def main():
    st.set_page_config(page_title="Batch Processing - ADAPT-SQL", page_icon="📦", layout="wide")
    
    st.title("📦 Batch Processing")
    st.markdown("Process multiple queries and analyze results in detail")
    st.markdown("---")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
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
        
        # Batch size selection
        num_queries = st.number_input("Number of Queries", min_value=1, max_value=100, value=5)
        start_idx = st.number_input("Start Index", min_value=0, value=0)
        
        st.markdown("---")
        
        if st.button("📂 Load Dataset"):
            try:
                with open(spider_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                st.session_state.spider_data = data
                st.success(f"✅ {len(data)} examples loaded")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    
    # Main content
    if 'spider_data' not in st.session_state:
        st.info("👈 Load dataset from sidebar to begin")
        return
    
    st.success(f"📊 Dataset loaded: {len(st.session_state.spider_data)} examples")
    
    # Batch processing section
    st.markdown("## 🚀 Batch Processing")
    
    end_idx = min(start_idx + num_queries, len(st.session_state.spider_data))
    st.info(f"Will process queries {start_idx} to {end_idx - 1} ({end_idx - start_idx} queries)")
    
    if st.button("▶️ Run Batch Processing", type="primary", use_container_width=True):
        # Initialize ADAPT
        adapt = ADAPTBaseline(model=model, vector_store_path=vector_store_path)
        
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
                st.warning(f"⚠️ Database not found for query {i}: {example['db_id']}")
                continue
            
            # Load schema
            schema_dict = get_schema_from_sqlite(str(db_path))
            foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
            
            # Run pipeline
            try:
                result = adapt.run_full_pipeline(
                    example['question'],
                    schema_dict,
                    foreign_keys,
                    k_examples=k_examples
                )
                
                results.append({
                    'index': i,
                    'example': example,
                    'result': result
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
        
        st.success(f"✅ Batch processing complete! Processed {len(results)} queries.")
    
    # Display results if available
    if 'batch_results' in st.session_state:
        st.markdown("---")
        st.markdown(f"## 📊 Results (Generated at {st.session_state.batch_timestamp})")
        
        results = st.session_state.batch_results
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", len(results))
        
        with col2:
            valid_count = sum(1 for r in results if r['result'].get('step7', {}).get('is_valid', False))
            st.metric("Valid SQL", valid_count)
        
        with col3:
            easy_count = sum(1 for r in results if r['result']['step2']['complexity_class'].value == "EASY")
            st.metric("EASY Queries", easy_count)
        
        with col4:
            avg_score = sum(r['result'].get('step7', {}).get('validation_score', 0) for r in results) / len(results) if results else 0
            st.metric("Avg Validation Score", f"{avg_score:.2f}")
        
        # Complexity distribution
        st.markdown("### 🎯 Complexity Distribution")
        complexity_counts = {}
        for r in results:
            complexity = r['result']['step2']['complexity_class'].value
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 EASY", complexity_counts.get("EASY", 0))
        with col2:
            st.metric("🟡 NON_NESTED_COMPLEX", complexity_counts.get("NON_NESTED_COMPLEX", 0))
        with col3:
            st.metric("🔴 NESTED_COMPLEX", complexity_counts.get("NESTED_COMPLEX", 0))
        
        st.markdown("---")
        
        # Detailed results for each query
        st.markdown("### 🔍 Detailed Query Results")
        
        # Filter options
        col1, col2 = st.columns(2)
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
        
        # Apply filters
        filtered_results = results
        
        if filter_complexity:
            filtered_results = [
                r for r in filtered_results 
                if r['result']['step2']['complexity_class'].value in filter_complexity
            ]
        
        if filter_validity == "Valid Only":
            filtered_results = [
                r for r in filtered_results 
                if r['result'].get('step7', {}).get('is_valid', False)
            ]
        elif filter_validity == "Invalid Only":
            filtered_results = [
                r for r in filtered_results 
                if not r['result'].get('step7', {}).get('is_valid', True)
            ]
        
        st.info(f"Showing {len(filtered_results)} of {len(results)} queries")
        
        # Display each query
        for r in filtered_results:
            display_query_details(r['index'], r['example'], r['result'])
        
        # Export options
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Summary CSV"):
                # Create summary dataframe
                summary_data = []
                for r in results:
                    generated_sql = None
                    if r['result'].get('step6a'):
                        generated_sql = r['result']['step6a']['generated_sql']
                    elif r['result'].get('step6b'):
                        generated_sql = r['result']['step6b']['generated_sql']
                    elif r['result'].get('step6c'):
                        generated_sql = r['result']['step6c']['generated_sql']
                    
                    summary_data.append({
                        'Index': r['index'],
                        'Database': r['example']['db_id'],
                        'Question': r['example']['question'],
                        'Complexity': r['result']['step2']['complexity_class'].value,
                        'Strategy': r['result']['step5']['strategy'].value,
                        'Generated_SQL': generated_sql,
                        'Ground_Truth_SQL': r['example'].get('query', ''),
                        'Is_Valid': r['result'].get('step7', {}).get('is_valid', None),
                        'Validation_Score': r['result'].get('step7', {}).get('validation_score', None),
                        'Num_Errors': len(r['result'].get('step7', {}).get('errors', [])),
                        'Num_Warnings': len(r['result'].get('step7', {}).get('warnings', []))
                    })
                
                df = pd.DataFrame(summary_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"adapt_sql_batch_results_{st.session_state.batch_timestamp.replace(' ', '_').replace(':', '-')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Export Full JSON"):
                # Prepare full results for JSON export
                export_data = []
                for r in results:
                    # Convert non-serializable objects
                    result_copy = {}
                    for key, value in r['result'].items():
                        if key in ['step2', 'step5']:
                            # Handle enums
                            if key == 'step2':
                                result_copy[key] = {
                                    'complexity_class': value['complexity_class'].value,
                                    'required_tables': list(value['required_tables']),
                                    'sub_questions': value['sub_questions'],
                                    'needs_joins': value['needs_joins'],
                                    'needs_subqueries': value['needs_subqueries'],
                                    'aggregations': value['aggregations'],
                                    'has_grouping': value['has_grouping'],
                                    'has_ordering': value['has_ordering']
                                }
                            elif key == 'step5':
                                result_copy[key] = {
                                    'strategy': value['strategy'].value,
                                    'reasoning': value['reasoning'],
                                    'description': value['description']
                                }
                        elif key == 'step1':
                            result_copy[key] = {
                                'schema_links': {
                                    'tables': list(value['schema_links']['tables']),
                                    'columns': {k: list(v) for k, v in value['schema_links']['columns'].items()},
                                    'foreign_keys': value['schema_links']['foreign_keys']
                                }
                            }
                        else:
                            result_copy[key] = value
                    
                    export_data.append({
                        'index': r['index'],
                        'example': r['example'],
                        'result': result_copy
                    })
                
                json_str = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"adapt_sql_batch_results_{st.session_state.batch_timestamp.replace(' ', '_').replace(':', '-')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()