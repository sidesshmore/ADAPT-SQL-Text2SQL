"""
ADAPT-SQL Streamlit Application - Steps 1-4
"""
import streamlit as st
import json
import sqlite3
from pathlib import Path
from adapt_baseline import ADAPTBaseline


# Page config
st.set_page_config(
    page_title="ADAPT-SQL: Steps 1-4",
    page_icon="🎯",
    layout="wide"
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
        st.error(f"Error loading Spider data: {e}")
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
        st.error(f"Error reading schema: {e}")
        return {}


def get_foreign_keys_from_sqlite(db_path: str) -> list:
    """Extract foreign keys from SQLite database"""
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
        st.error(f"Error reading foreign keys: {e}")
        return []


def display_step1_tab(step1):
    """Display Step 1 results"""
    st.markdown("### 📊 Schema Linking Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tables", len(step1['schema_links']['tables']))
    
    with col2:
        total_cols = sum(len(cols) for cols in step1['schema_links']['columns'].values())
        st.metric("Columns", total_cols)
    
    with col3:
        st.metric("Foreign Keys", len(step1['schema_links']['foreign_keys']))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Relevant Tables**")
        for table in sorted(step1['schema_links']['tables']):
            st.success(f"📊 {table}")
    
    with col2:
        st.markdown("**✅ Relevant Columns**")
        for table, cols in sorted(step1['schema_links']['columns'].items()):
            with st.expander(f"📋 {table}"):
                for col in sorted(cols):
                    st.text(f"  • {col}")
    
    with st.expander("🧠 Reasoning"):
        st.text(step1['reasoning'])


def display_step2_tab(step2):
    """Display Step 2 results"""
    st.markdown("### 🔍 Complexity Classification Results")
    
    complexity = step2['complexity_class'].value
    
    if complexity == "EASY":
        st.success(f"## 🟢 {complexity}")
    elif complexity == "NON_NESTED_COMPLEX":
        st.warning(f"## 🟡 {complexity}")
    else:
        st.error(f"## 🔴 {complexity}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Requirements**")
        st.write(f"• Tables: {len(step2['required_tables'])}")
        st.write(f"• JOINs: {'✅' if step2['needs_joins'] else '❌'}")
        st.write(f"• Subqueries: {'✅' if step2['needs_subqueries'] else '❌'}")
    
    with col2:
        st.markdown("**🔧 Operations**")
        if step2['aggregations']:
            st.write(f"• Aggregations: {', '.join(step2['aggregations'])}")
        st.write(f"• GROUP BY: {'✅' if step2['has_grouping'] else '❌'}")
        st.write(f"• ORDER BY: {'✅' if step2['has_ordering'] else '❌'}")
    
    if step2['sub_questions']:
        st.markdown("---")
        st.markdown(f"**🔎 Sub-Questions ({len(step2['sub_questions'])})**")
        for i, sq in enumerate(step2['sub_questions'], 1):
            st.write(f"{i}. {sq}")
    
    with st.expander("🧠 Reasoning"):
        st.text(step2['reasoning'])


def display_step3_tab(step3):
    """Display Step 3 results"""
    st.markdown("### 💻 Preliminary SQL Prediction")
    
    st.markdown("**🎯 Predicted SQL:**")
    st.code(step3['predicted_sql'], language='sql')
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**🗝️ SQL Skeleton:**")
        st.info(step3['sql_skeleton'])
    
    with col2:
        st.metric("Complexity Score", step3['sql_structure']['complexity_score'])
    
    with st.expander("🧠 Reasoning"):
        st.text(step3['reasoning'])


def display_step4_tab(step4):
    """Display Step 4 results - Similarity Search"""
    st.markdown("### 🔎 Similar Examples from Vector Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Query", step4['query'][:50] + "..." if len(step4['query']) > 50 else step4['query'])
    
    with col2:
        st.metric("Similar Examples Found", step4['total_found'])
    
    st.markdown("---")
    
    st.markdown(f"**📊 Top Similar Examples (Ranked by Similarity Score):**")
    
    if not step4['similar_examples']:
        st.warning("No similar examples found in vector database")
    else:
        for i, example in enumerate(step4['similar_examples'], 1):
            similarity_score = example.get('similarity_score', 0)
            
            # Color code based on similarity score
            if similarity_score >= 0.8:
                score_color = "🟢"
            elif similarity_score >= 0.6:
                score_color = "🟡"
            else:
                score_color = "🔴"
            
            with st.expander(f"{score_color} Example {i}: {example.get('question', 'N/A')[:70]}... (Score: {similarity_score:.4f})"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown("**📊 Metadata**")
                    st.text(f"Database: {example.get('db_id', 'unknown')}")
                    st.text(f"Similarity: {similarity_score:.4f}")
                
                with col2:
                    st.markdown("**❓ Question:**")
                    st.info(example.get('question', 'N/A'))
                
                st.markdown("**💾 Gold SQL Query:**")
                st.code(example.get('query', 'N/A'), language='sql')
    
    with st.expander("🧠 Reasoning"):
        st.text(step4['reasoning'])


def main():
    st.title("🎯 ADAPT-SQL: Steps 1-4")
    st.markdown("Schema → Complexity → SQL → Similarity Search")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model = st.selectbox(
            "Ollama Model",
            ["llama3.2", "codellama", "mistral", "qwen2.5"]
        )
        
        spider_json_path = st.text_input(
            "Spider dev.json",
            value="/Users/sidessh/ADAPT-SQL/data/spider/dev.json"
        )
        
        spider_db_dir = st.text_input(
            "Spider DB directory",
            value="/Users/sidessh/ADAPT-SQL/data/spider/spider_data/database"
        )
        
        vector_store_path = st.text_input(
            "Vector Store Path",
            value="./vector_store"
        )
        
        k_examples = st.slider("Similar Examples to Retrieve", 1, 20, 10)
        
        st.markdown("---")
        
        if st.button("📂 Load Spider Dataset"):
            data = load_spider_data(spider_json_path)
            if data:
                st.session_state.spider_data = data
                st.success(f"✅ Loaded {len(data)} examples")
        
        if st.session_state.spider_data:
            st.info(f"📊 {len(st.session_state.spider_data)} examples loaded")
    
    # Main content
    if not st.session_state.spider_data:
        st.info("👈 Please load Spider dataset from sidebar")
        return
    
    # Example selection
    st.header("🔍 Select Question")
    
    example_idx = st.selectbox(
        "Spider Example",
        range(len(st.session_state.spider_data)),
        format_func=lambda i: f"Ex {i+1}: {st.session_state.spider_data[i]['question'][:80]}..."
    )
    
    example = st.session_state.spider_data[example_idx]
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📋 Question")
        st.info(example['question'])
    with col2:
        st.markdown("### 🗄️ Database")
        st.code(example['db_id'])
    
    if 'query' in example:
        with st.expander("🎯 Ground Truth SQL"):
            st.code(example['query'], language='sql')
    
    st.markdown("---")
    
    # Run ADAPT
    if st.button("🚀 Run ADAPT Steps 1-4", type="primary", use_container_width=True):
        with st.spinner("🔄 Processing..."):
            # Get database
            db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
            
            if not db_path.exists():
                st.error(f"❌ Database not found: {db_path}")
                return
            
            # Get schema and FKs
            schema_dict = get_schema_from_sqlite(str(db_path))
            foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
            
            # Initialize ADAPT
            adapt = ADAPTBaseline(
                model=model,
                vector_store_path=vector_store_path
            )
            
            # Run all steps
            result = adapt.run_steps_1_to_4(
                example['question'],
                schema_dict,
                foreign_keys,
                k_examples=k_examples
            )
            
            st.success("✅ All Steps Complete!")
            st.markdown("---")
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 STEP 1: Schema", "🔍 STEP 2: Complexity", "💻 STEP 3: SQL", "🔎 STEP 4: Similar"
            ])
            
            with tab1:
                display_step1_tab(result['step1'])
            
            with tab2:
                display_step2_tab(result['step2'])
            
            with tab3:
                display_step3_tab(result['step3'])
            
            with tab4:
                display_step4_tab(result['step4'])


if __name__ == "__main__":
    main()