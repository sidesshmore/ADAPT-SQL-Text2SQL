"""
ADAPT-SQL Streamlit Application - Steps 1, 2 & 3
Schema Linking + Complexity Classification + Preliminary SQL Prediction
"""
import streamlit as st
import json
import sqlite3
from pathlib import Path
from adapt_baseline import ADAPTBaseline
from prel_sql_prediction import PreliminaryPredictor


# Page config
st.set_page_config(
    page_title="ADAPT-SQL: Steps 1-3",
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
            data = json.load(f)
        return data
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


def main():
    st.title("🎯 ADAPT-SQL: Schema Linking + Complexity + SQL Prediction")
    st.markdown("### Steps 1-3: Schema → Complexity → Preliminary SQL")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model = st.selectbox(
            "Ollama Model",
            ["llama3.2", "codellama", "mistral", "qwen2.5"],
            help="Select the Ollama model to use"
        )
        
        spider_json_path = st.text_input(
            "Spider dev.json path",
            value="/Users/sidessh/ADAPT-SQL/data/spider/dev.json"
        )
        
        spider_db_dir = st.text_input(
            "Spider database directory",
            value="/Users/sidessh/ADAPT-SQL/data/spider/spider_data/database"
        )
        
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
        st.info("👈 Please load Spider dataset from the sidebar to begin")
        return
    
    # Example selection
    st.header("🔍 Select Question")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        example_idx = st.selectbox(
            "Spider Example",
            range(len(st.session_state.spider_data)),
            format_func=lambda i: f"Example {i+1}: {st.session_state.spider_data[i]['question'][:80]}..."
        )
    
    with col2:
        if st.button("🎲 Random Example"):
            import random
            example_idx = random.randint(0, len(st.session_state.spider_data) - 1)
            st.rerun()
    
    example = st.session_state.spider_data[example_idx]
    
    # Display question
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📋 Question")
        st.info(example['question'])
    with col2:
        st.markdown("### 🗄️ Database")
        st.code(example['db_id'])
    
    # Show ground truth SQL if available
    if 'query' in example:
        with st.expander("🎯 Ground Truth SQL"):
            st.code(example['query'], language='sql')
    
    # Show full schema
    with st.expander("🔍 View Full Database Schema"):
        db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
        
        if db_path.exists():
            full_schema = get_schema_from_sqlite(str(db_path))
            foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
            
            st.markdown(f"**{len(full_schema)} Tables:**")
            for table, cols in full_schema.items():
                with st.expander(f"📊 {table} ({len(cols)} columns)"):
                    for col in cols:
                        st.text(f"  • {col['column_name']}: {col['data_type']}")
            
            if foreign_keys:
                st.markdown(f"**{len(foreign_keys)} Foreign Keys:**")
                for fk in foreign_keys:
                    st.text(f"  → {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}")
    
    st.markdown("---")
    
    # Generate button
    if st.button("🚀 Run ADAPT Steps 1-3", type="primary", use_container_width=True):
        with st.spinner("🔄 Running schema linking, complexity classification, and SQL prediction..."):
            # Get database
            db_path = Path(spider_db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
            
            if not db_path.exists():
                st.error(f"❌ Database not found: {db_path}")
                return
            
            # Get schema and FKs
            schema_dict = get_schema_from_sqlite(str(db_path))
            foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
            
            # Initialize ADAPT
            adapt = ADAPTBaseline(model=model)
            predictor = PreliminaryPredictor(model=model)
            
            # Run Steps 1 & 2
            result = adapt.run_steps_1_and_2(
                example['question'],
                schema_dict,
                foreign_keys
            )
            
            step1 = result['step1']
            step2 = result['step2']
            
            # Run Step 3
            step3 = predictor.predict_sql_skeleton(
                example['question'],
                step1['pruned_schema'],
                step1['schema_links']
            )
            
            st.success("✅ All Steps Complete!")
            st.markdown("---")
            
            # Create tabs for all 3 steps
            tab1, tab2, tab3 = st.tabs([
                "📊 STEP 1: Schema Linking", 
                "🔍 STEP 2: Complexity Classification",
                "💻 STEP 3: Preliminary SQL"
            ])
            
            # ============= STEP 1 TAB =============
            with tab1:
                st.markdown("### 📊 Schema Linking Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Tables",
                        len(step1['schema_links']['tables']),
                        delta=f"-{len(schema_dict) - len(step1['schema_links']['tables'])}"
                    )
                
                with col2:
                    total_cols = sum(len(cols) for cols in step1['schema_links']['columns'].values())
                    original_cols = sum(len(cols) for cols in schema_dict.values())
                    st.metric(
                        "Columns",
                        total_cols,
                        delta=f"-{original_cols - total_cols}"
                    )
                
                with col3:
                    st.metric(
                        "Foreign Keys",
                        len(step1['schema_links']['foreign_keys'])
                    )
                
                with col4:
                    st.metric(
                        "JOIN Paths",
                        len(step1['schema_links']['join_paths'])
                    )
                
                st.markdown("---")
                
                # Tables and columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Relevant Tables**")
                    if step1['schema_links']['tables']:
                        for table in sorted(step1['schema_links']['tables']):
                            st.success(f"📊 {table}")
                    else:
                        st.info("No tables identified")
                
                with col2:
                    st.markdown("**✅ Relevant Columns**")
                    if step1['schema_links']['columns']:
                        for table, cols in sorted(step1['schema_links']['columns'].items()):
                            with st.expander(f"📋 {table}"):
                                for col in sorted(cols):
                                    st.text(f"  • {col}")
                    else:
                        st.info("No columns identified")
                
                # Foreign keys
                if step1['schema_links']['foreign_keys']:
                    st.markdown("**🔗 Critical Foreign Keys**")
                    for fk in step1['schema_links']['foreign_keys']:
                        st.info(f"→ {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}")
                
                # JOIN paths
                if step1['schema_links']['join_paths']:
                    st.markdown("**🛣️ JOIN Paths**")
                    for i, path in enumerate(step1['schema_links']['join_paths'], 1):
                        st.success(f"Path {i}: {' → '.join(path)}")
                
                st.markdown("---")
                
                # Pruned schema
                st.markdown("**📦 Pruned Schema**")
                with st.expander("View Pruned Schema"):
                    for table_name, columns in sorted(step1['pruned_schema'].items()):
                        st.markdown(f"**{table_name}**")
                        for col in columns:
                            st.text(f"  • {col['column_name']}: {col['data_type']}")
                        st.markdown("")
                
                # Reasoning
                with st.expander("🧠 Step 1 Reasoning"):
                    st.text(step1['reasoning'])
            
            # ============= STEP 2 TAB =============
            with tab2:
                st.markdown("### 🔍 Complexity Classification Results")
                
                # Complexity class (big display)
                complexity = step2['complexity_class'].value
                
                if complexity == "EASY":
                    st.success(f"## 🟢 {complexity}")
                    st.info("Single table or simple JOIN, no subqueries")
                elif complexity == "NON_NESTED_COMPLEX":
                    st.warning(f"## 🟡 {complexity}")
                    st.info("Multiple JOINs, no subqueries")
                else:  # NESTED_COMPLEX
                    st.error(f"## 🔴 {complexity}")
                    st.info("Requires subqueries or nested SELECT")
                
                st.markdown("---")
                
                # Analysis details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📋 Query Requirements**")
                    st.write(f"• **Tables:** {len(step2['required_tables'])}")
                    st.write(f"• **JOINs needed:** {'✅ Yes' if step2['needs_joins'] else '❌ No'}")
                    st.write(f"• **Subqueries needed:** {'✅ Yes' if step2['needs_subqueries'] else '❌ No'}")
                    st.write(f"• **Set operations:** {'✅ Yes' if step2['needs_set_operations'] else '❌ No'}")
                
                with col2:
                    st.markdown("**🔧 SQL Operations**")
                    if step2['aggregations']:
                        st.write(f"• **Aggregations:** {', '.join(step2['aggregations'])}")
                    else:
                        st.write("• **Aggregations:** None")
                    st.write(f"• **GROUP BY:** {'✅ Yes' if step2['has_grouping'] else '❌ No'}")
                    st.write(f"• **ORDER BY:** {'✅ Yes' if step2['has_ordering'] else '❌ No'}")
                
                st.markdown("---")
                
                # Generation strategy
                from query_complexity import QueryComplexityClassifier
                classifier = QueryComplexityClassifier()
                strategy = classifier.get_generation_strategy(step2['complexity_class'])
                
                st.markdown("**🎯 Recommended Generation Strategy**")
                st.info(f"**{strategy}**")
                
                if strategy == "SIMPLE_FEW_SHOT":
                    st.write("→ Use simple few-shot examples to generate SQL directly")
                elif strategy == "INTERMEDIATE_REPRESENTATION":
                    st.write("→ Use intermediate representation with explicit JOIN planning")
                else:  # DECOMPOSED_GENERATION
                    st.write("→ Decompose into sub-questions and generate incrementally")
                
                # Sub-questions
                if step2['sub_questions']:
                    st.markdown("---")
                    st.markdown(f"**🔍 Identified Sub-Questions ({len(step2['sub_questions'])})**")
                    for i, sq in enumerate(step2['sub_questions'], 1):
                        st.write(f"{i}. {sq}")
                
                # Structural hints
                st.markdown("---")
                st.markdown("**💡 Structural Hints**")
                hints = step2['structural_hints']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"• Has aggregation: {hints['has_aggregation']}")
                    st.write(f"• Has comparison: {hints['has_comparison']}")
                
                with col2:
                    st.write(f"• Has superlative: {hints['has_superlative']}")
                    st.write(f"• Nested logic: {hints['has_nested_logic']}")
                
                # Reasoning
                st.markdown("---")
                with st.expander("🧠 Step 2 Reasoning"):
                    st.text(step2['reasoning'])
            
            # ============= STEP 3 TAB =============
            with tab3:
                st.markdown("### 💻 Preliminary SQL Prediction Results")
                
                # Display predicted SQL prominently
                st.markdown("**🎯 Predicted SQL Query:**")
                st.code(step3['predicted_sql'], language='sql')
                
                st.markdown("---")
                
                # SQL Skeleton
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**🏗️ SQL Skeleton:**")
                    st.info(step3['sql_skeleton'])
                
                with col2:
                    st.metric(
                        "Complexity Score",
                        step3['sql_structure']['complexity_score']
                    )
                
                st.markdown("---")
                
                # Structure analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Query Structure**")
                    struct = step3['sql_structure']
                    st.write(f"• **Type:** {struct['query_type']}")
                    st.write(f"• **SELECT statements:** {struct['num_selects']}")
                    st.write(f"• **JOINs:** {struct['num_joins']}")
                    st.write(f"• **Tables involved:** {struct['num_tables']}")
                    st.write(f"• **Has subquery:** {'✅ Yes' if struct['has_subquery'] else '❌ No'}")
                
                with col2:
                    st.markdown("**🔧 SQL Features**")
                    st.write(f"• **Aggregation:** {'✅ Yes' if struct['has_aggregation'] else '❌ No'}")
                    st.write(f"• **GROUP BY:** {'✅ Yes' if struct['has_groupby'] else '❌ No'}")
                    st.write(f"• **HAVING:** {'✅ Yes' if struct['has_having'] else '❌ No'}")
                    st.write(f"• **ORDER BY:** {'✅ Yes' if struct['has_orderby'] else '❌ No'}")
                    st.write(f"• **DISTINCT:** {'✅ Yes' if struct['has_distinct'] else '❌ No'}")
                
                st.markdown("---")
                
                # SQL Keywords
                st.markdown(f"**🔑 SQL Keywords ({len(step3['sql_keywords'])}):**")
                keyword_cols = st.columns(4)
                for i, keyword in enumerate(step3['sql_keywords']):
                    with keyword_cols[i % 4]:
                        st.text(f"• {keyword}")
                
                st.markdown("---")
                
                # Comparison with ground truth if available
                if 'query' in example:
                    st.markdown("**⚖️ Comparison with Ground Truth**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("*Predicted:*")
                        st.code(step3['predicted_sql'], language='sql')
                    
                    with col2:
                        st.markdown("*Ground Truth:*")
                        st.code(example['query'], language='sql')
                    
                    # Basic similarity check
                    pred_skeleton = step3['sql_skeleton']
                    gt_predictor = PreliminaryPredictor(model=model)
                    gt_analysis = gt_predictor._extract_sql_skeleton(example['query'])
                    
                    if pred_skeleton == gt_analysis:
                        st.success("✅ SQL Skeleton matches ground truth!")
                    else:
                        st.warning(f"⚠️ Skeleton differs: Predicted `{pred_skeleton}` vs Ground Truth `{gt_analysis}`")
                
                st.markdown("---")
                
                # Reasoning
                with st.expander("🧠 Step 3 Reasoning"):
                    st.text(step3['reasoning'])
            
            # Download options
            st.markdown("---")
            st.markdown("### 📥 Download Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                import json
                schema_json = json.dumps(step1['pruned_schema'], indent=2)
                st.download_button(
                    "📦 Pruned Schema",
                    schema_json,
                    f"pruned_schema_{example['db_id']}.json",
                    "application/json"
                )
            
            with col2:
                st.download_button(
                    "🔍 Step 1 Reasoning",
                    step1['reasoning'],
                    f"step1_{example['db_id']}.txt",
                    "text/plain"
                )
            
            with col3:
                st.download_button(
                    "📋 Step 2 Reasoning",
                    step2['reasoning'],
                    f"step2_{example['db_id']}.txt",
                    "text/plain"
                )
            
            with col4:
                st.download_button(
                    "💻 Predicted SQL",
                    step3['predicted_sql'],
                    f"predicted_sql_{example['db_id']}.sql",
                    "text/plain"
                )


if __name__ == "__main__":
    main()