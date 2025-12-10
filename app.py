"""
ADAPT-SQL Streamlit Application - Updated with Step 6b
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


def main():
    st.title("🎯 ADAPT-SQL Pipeline")
    st.markdown("End-to-end Text-to-SQL Generation with Intermediate Representation")
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
    
    # Single Query Mode
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
            
            # Display results in tabs
            tabs = st.tabs([
                "📊 Schema", "🔍 Complexity", "🔎 Examples", "🔀 Route", "✨ SQL"
            ])
            
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
                
                # Step 6a: Simple Few-Shot
                if result.get('step6a'):
                    st.markdown("**Method:** Simple Few-Shot")
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
                
                # Step 6b: Intermediate Representation
                elif result.get('step6b'):
                    st.markdown("**Method:** Intermediate Representation")
                    
                    with st.expander("🔍 NatSQL Intermediate Representation"):
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
                
                else:
                    st.warning(f"⚠️ {result['step5']['strategy'].value} not implemented yet")


if __name__ == "__main__":
    main()