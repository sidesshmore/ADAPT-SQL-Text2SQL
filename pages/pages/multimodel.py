"""
ADAPT-SQL Multi-Model Comparison
Compare performance across multiple LLM models
"""
import streamlit as st
import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from adapt_baseline import ADAPTBaseline
from batch_utils import save_checkpoint, load_checkpoint


st.set_page_config(
    page_title="Multi-Model Comparison - ADAPT-SQL",
    page_icon="🔬",
    layout="wide"
)


# Available models
AVAILABLE_MODELS = [
    "llama3.2",
    "qwen3-coder",
    "gemma3",
    "codellama",
    "deepseek-r1:1.5b"
]


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


def display_model_comparison_statistics(comparison_results: dict):
    """Display comprehensive comparison statistics across models"""
    
    st.markdown("## 🔬 Multi-Model Performance Comparison")
    st.markdown("---")
    
    # Extract per-model results
    model_stats = {}
    
    for model_name, results in comparison_results.items():
        if not results:
            continue
        
        total = len(results)
        valid_count = sum(1 for r in results if r['result'].get('final_is_valid', False))
        
        stats = {
            'total': total,
            'valid': valid_count,
            'valid_rate': (valid_count / total * 100) if total > 0 else 0
        }
        
        # Execution stats
        if any(r['result'].get('step10_generated') for r in results):
            exec_success = sum(1 for r in results if r['result'].get('step10_generated', {}).get('success', False))
            stats['exec_success'] = exec_success
            stats['exec_rate'] = (exec_success / total * 100) if total > 0 else 0
            
            # Average execution time
            exec_times = [r['result']['step10_generated']['execution_time'] 
                         for r in results if r['result'].get('step10_generated', {}).get('success', False)]
            stats['avg_exec_time'] = sum(exec_times) / len(exec_times) if exec_times else 0
        
        # Evaluation stats
        if any(r['result'].get('step11') for r in results):
            ex_acc = sum(1 for r in results if r['result'].get('step11', {}).get('execution_accuracy', False))
            em_match = sum(1 for r in results if r['result'].get('step11', {}).get('exact_set_match', False))
            eval_results = [r for r in results if r['result'].get('step11')]
            avg_score = sum(r['result']['step11']['evaluation_score'] for r in eval_results) / len(eval_results) if eval_results else 0
            
            stats['ex_acc'] = ex_acc
            stats['ex_rate'] = (ex_acc / total * 100) if total > 0 else 0
            stats['em_match'] = em_match
            stats['em_rate'] = (em_match / total * 100) if total > 0 else 0
            stats['avg_score'] = avg_score
        
        model_stats[model_name] = stats
    
    # =====================================================================
    # OVERVIEW TABLE
    # =====================================================================
    st.markdown("### 📊 Overview Table")
    
    overview_data = []
    for model_name, stats in sorted(model_stats.items()):
        row = {
            'Model': model_name,
            'Valid SQL (%)': f"{stats['valid_rate']:.1f}",
        }
        
        if 'exec_rate' in stats:
            row['Execution (%)'] = f"{stats['exec_rate']:.1f}"
            row['Avg Time (s)'] = f"{stats['avg_exec_time']:.3f}"
        
        if 'ex_rate' in stats:
            row['EX = 1.0 (%)'] = f"{stats['ex_rate']:.1f}"
            row['EM = 1.0 (%)'] = f"{stats['em_rate']:.1f}"
            row['Avg Score'] = f"{stats['avg_score']:.3f}"
        
        overview_data.append(row)
    
    if overview_data:
        df = pd.DataFrame(overview_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # =====================================================================
    # KEY METRICS COMPARISON
    # =====================================================================
    st.markdown("### 🎯 Key Metrics Comparison")
    
    cols = st.columns(len(model_stats))
    
    for i, (model_name, stats) in enumerate(sorted(model_stats.items())):
        with cols[i]:
            st.markdown(f"**{model_name}**")
            
            # Valid SQL
            st.metric(
                "Valid SQL",
                f"{stats['valid_rate']:.1f}%",
                f"{stats['valid']}/{stats['total']}"
            )
            
            # Execution
            if 'exec_rate' in stats:
                st.metric(
                    "Execution",
                    f"{stats['exec_rate']:.1f}%",
                    f"{stats['exec_success']}/{stats['total']}"
                )
            
            # EX = 1.0
            if 'ex_rate' in stats:
                st.metric(
                    "EX = 1.0",
                    f"{stats['ex_rate']:.1f}%",
                    f"{stats['ex_acc']}/{stats['total']}"
                )
            
            # Average Score
            if 'avg_score' in stats:
                st.metric(
                    "Avg Score",
                    f"{stats['avg_score']:.3f}"
                )
    
    st.markdown("---")
    
    # =====================================================================
    # RANKING
    # =====================================================================
    st.markdown("### 🏆 Model Rankings")
    
    # Rank by EX accuracy
    if all('ex_rate' in stats for stats in model_stats.values()):
        st.markdown("**By Execution Accuracy (EX = 1.0):**")
        
        ranked = sorted(model_stats.items(), key=lambda x: x[1]['ex_rate'], reverse=True)
        
        for rank, (model_name, stats) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            st.write(f"{medal} **{model_name}**: {stats['ex_rate']:.1f}% ({stats['ex_acc']}/{stats['total']})")
    
    st.markdown("")
    
    # Rank by average score
    if all('avg_score' in stats for stats in model_stats.values()):
        st.markdown("**By Average Composite Score:**")
        
        ranked = sorted(model_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        
        for rank, (model_name, stats) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            st.write(f"{medal} **{model_name}**: {stats['avg_score']:.3f}")
    
    st.markdown("---")
    
    # =====================================================================
    # COMPLEXITY BREAKDOWN
    # =====================================================================
    st.markdown("### 📈 Performance by Query Complexity")
    
    # Aggregate by complexity
    complexity_breakdown = {}
    
    for model_name, results in comparison_results.items():
        if not results:
            continue
        
        for complexity in ["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"]:
            complexity_results = [r for r in results if r['result']['step2']['complexity_class'].value == complexity]
            
            if not complexity_results:
                continue
            
            count = len(complexity_results)
            valid = sum(1 for r in complexity_results if r['result'].get('final_is_valid', False))
            
            if complexity not in complexity_breakdown:
                complexity_breakdown[complexity] = {}
            
            stats = {
                'count': count,
                'valid': valid,
                'valid_rate': (valid / count * 100) if count > 0 else 0
            }
            
            if any(r['result'].get('step11') for r in complexity_results):
                ex_acc = sum(1 for r in complexity_results if r['result'].get('step11', {}).get('execution_accuracy', False))
                eval_count = len([r for r in complexity_results if r['result'].get('step11')])
                stats['ex_acc'] = ex_acc
                stats['ex_rate'] = (ex_acc / eval_count * 100) if eval_count > 0 else 0
            
            complexity_breakdown[complexity][model_name] = stats
    
    # Display breakdown
    for complexity, model_data in complexity_breakdown.items():
        with st.expander(f"**{complexity}** ({len(model_data)} models)"):
            cols = st.columns(len(model_data))
            
            for i, (model_name, stats) in enumerate(sorted(model_data.items())):
                with cols[i]:
                    st.markdown(f"**{model_name}**")
                    st.write(f"Valid: {stats['valid']}/{stats['count']} ({stats['valid_rate']:.1f}%)")
                    
                    if 'ex_rate' in stats:
                        st.write(f"EX=1.0: {stats['ex_acc']} ({stats['ex_rate']:.1f}%)")


def export_comparison_csv(comparison_results: dict, timestamp: str, output_dir: Path) -> Path:
    """Export comparison results to CSV"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison data
    comparison_data = []
    
    # Get all queries (from first model)
    first_model = list(comparison_results.keys())[0]
    first_results = comparison_results[first_model]
    
    for i, result_data in enumerate(first_results):
        example = result_data['example']
        
        row = {
            'Index': i,
            'Database': example['db_id'],
            'Question': example['question'],
            'Ground_Truth_SQL': example.get('query', ''),
            'Complexity': result_data['result']['step2']['complexity_class'].value
        }
        
        # Add per-model results
        for model_name, model_results in comparison_results.items():
            if i >= len(model_results):
                continue
            
            result = model_results[i]['result']
            
            # SQL
            row[f'{model_name}_SQL'] = result.get('final_sql', '')
            
            # Valid
            row[f'{model_name}_Valid'] = result.get('final_is_valid', False)
            
            # Execution
            if result.get('step10_generated'):
                row[f'{model_name}_Exec_Success'] = result['step10_generated']['success']
                row[f'{model_name}_Exec_Time'] = result['step10_generated']['execution_time']
            
            # Evaluation
            if result.get('step11'):
                row[f'{model_name}_EX'] = result['step11']['execution_accuracy']
                row[f'{model_name}_EM'] = result['step11']['exact_set_match']
                row[f'{model_name}_Score'] = result['step11']['evaluation_score']
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_filename = f"multimodel_comparison_{timestamp.replace(' ', '_').replace(':', '-')}.csv"
    csv_path = output_dir / csv_filename
    
    df.to_csv(csv_path, index=False)
    
    return csv_path


def main():
    st.title("🔬 ADAPT-SQL Multi-Model Comparison")
    st.markdown("Compare performance across multiple LLM models on the same queries")
    st.markdown("---")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        
        selected_models = []
        for model in AVAILABLE_MODELS:
            if st.checkbox(model, value=True):
                selected_models.append(model)
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model")
        else:
            st.success(f"✅ {len(selected_models)} models selected")
        
        st.markdown("---")
        
        # Dataset configuration
        st.markdown("### 📁 Dataset Configuration")
        
        spider_json_path = st.text_input(
            "📄 Spider dev.json",
            value="/home/smore123/ADAPT-SQL/data/spider/dev.json"
        )
        
        spider_db_dir = st.text_input(
            "📂 Spider DB directory",
            value="/home/smore123/ADAPT-SQL/data/spider/spider_data/database"
        )
        
        vector_store_path = st.text_input(
            "📚 Vector Store",
            value="./vector_store"
        )
        
        st.markdown("---")
        
        # Query settings
        st.markdown("### 🎯 Query Settings")
        
        num_queries = st.number_input("Number of Queries", min_value=1, max_value=100, value=10, step=5)
        start_idx = st.number_input("Start Index", min_value=0, value=0)
        k_examples = st.slider("📖 Similar Examples", 1, 20, 10)
        
        st.markdown("---")
        
        # Processing options
        st.markdown("### 🔧 Processing Options")
        
        enable_validation_retry = st.checkbox("✅ Validation Retry (Step 8)", value=True)
        enable_execution = st.checkbox("⚡ SQL Execution (Step 10)", value=True)
        enable_evaluation = st.checkbox("📊 Evaluation (Step 11)", value=True)
        
        st.markdown("---")
        
        # Output settings
        st.markdown("### 💾 Output Settings")
        
        results_dir = "./multimodel_results"
        st.info(f"📂 Results: `{results_dir}`")
        
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
        
        1. **Select Models** (sidebar)
           - Choose 2+ models to compare
           - Default: All 5 models selected
        
        2. **Configure Paths** (sidebar)
           - Set Spider dev.json path
           - Set database directory
           - Set vector store path
        
        3. **Load Dataset**
           - Click "Load Dataset" button
        
        4. **Configure Queries**
           - Set number of queries (recommended: 10-50)
           - Set start index
        
        5. **Run Comparison**
           - Click "Run Multi-Model Comparison"
           - Results auto-exported to CSV
        
        ### 📊 Comparison Metrics:
        
        - **Valid SQL Rate**: Percentage of syntactically valid queries
        - **Execution Success**: Percentage that execute without errors
        - **Execution Accuracy (EX)**: Primary Spider metric - results match exactly
        - **Exact-Set-Match (EM)**: Secondary Spider metric - SQL structure matches
        - **Composite Score**: Weighted (80% EX + 20% EM)
        - **Average Execution Time**: Query performance
        
        ### 🏆 Rankings:
        
        - Models ranked by EX accuracy (primary)
        - Models ranked by composite score
        - Breakdown by query complexity (EASY/NON_NESTED_COMPLEX/NESTED_COMPLEX)
        """)
        
        return
    
    # Check if models are selected
    if not selected_models:
        st.warning("⚠️ Please select at least one model from the sidebar")
        return
    
    # Display configuration
    st.markdown("## 🎯 Comparison Configuration")
    
    end_idx = min(start_idx + num_queries, len(st.session_state.spider_data))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models", len(selected_models))
    with col2:
        st.metric("Queries", end_idx - start_idx)
    with col3:
        st.metric("Start Index", start_idx)
    with col4:
        st.metric("Total Runs", len(selected_models) * (end_idx - start_idx))
    
    st.markdown("**Selected Models:**")
    st.write(", ".join(selected_models))
    
    st.markdown("---")
    
    if st.button("🚀 Run Multi-Model Comparison", type="primary", use_container_width=True):
        # Create output directory
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        comparison_results = {model: [] for model in selected_models}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Overall progress
        total_runs = len(selected_models) * (end_idx - start_idx)
        overall_progress = st.progress(0)
        overall_status = st.empty()
        
        current_run = 0
        
        # Process each model
        for model_idx, model_name in enumerate(selected_models):
            st.markdown(f"### 🤖 Processing with {model_name}")
            st.markdown(f"**Model {model_idx + 1}/{len(selected_models)}**")
            
            # Initialize ADAPT for this model
            with st.spinner(f"Initializing {model_name}..."):
                try:
                    adapt = ADAPTBaseline(model=model_name, vector_store_path=vector_store_path)
                    st.success(f"✅ {model_name} initialized")
                except Exception as e:
                    st.error(f"❌ Failed to initialize {model_name}: {e}")
                    continue
            
            # Model-specific progress
            model_progress = st.progress(0)
            model_status = st.empty()
            
            # Process each query
            for i in range(start_idx, end_idx):
                example = st.session_state.spider_data[i]
                
                # Update progress
                query_num = i - start_idx + 1
                model_progress.progress(query_num / (end_idx - start_idx))
                model_status.markdown(f"**Query {query_num}/{end_idx - start_idx}:** {example['question'][:60]}...")
                
                current_run += 1
                overall_progress.progress(current_run / total_runs)
                overall_status.markdown(f"**Overall Progress:** {current_run}/{total_runs} runs ({model_name})")
                
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
                    
                    comparison_results[model_name].append({
                        'index': i,
                        'example': example,
                        'result': result
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error processing query {i} with {model_name}: {str(e)}")
                    continue
            
            model_progress.empty()
            model_status.empty()
            
            # Show quick stats for this model
            results = comparison_results[model_name]
            valid_count = sum(1 for r in results if r['result'].get('final_is_valid', False))
            st.info(f"✅ {model_name} completed: {valid_count}/{len(results)} valid SQL queries")
            
            st.markdown("---")
        
        overall_progress.empty()
        overall_status.empty()
        
        # Store results in session state
        st.session_state.comparison_results = comparison_results
        st.session_state.comparison_timestamp = timestamp
        
        # Export to CSV
        st.markdown("### 💾 Exporting Results...")
        
        with st.spinner("Creating comparison CSV..."):
            csv_path = export_comparison_csv(
                comparison_results,
                timestamp,
                results_path
            )
        
        st.success(f"✅ Multi-model comparison complete!")
        st.success(f"📊 CSV saved to: **{csv_path}**")
        
        # Offer download button
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_data = f.read()
        
        st.download_button(
            label="⬇️ Download Comparison CSV",
            data=csv_data,
            file_name=csv_path.name,
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
    
    # Display comparison results if available
    if 'comparison_results' in st.session_state:
        comparison_results = st.session_state.comparison_results
        
        st.markdown("---")
        
        # Display comprehensive comparison statistics
        display_model_comparison_statistics(comparison_results)
        
        st.markdown("---")
        
        # Detailed per-query comparison
        with st.expander("🔍 View Detailed Per-Query Comparison"):
            st.markdown("### 📋 Per-Query Results")
            
            # Get queries from first model
            first_model = list(comparison_results.keys())[0]
            first_results = comparison_results[first_model]
            
            for i, result_data in enumerate(first_results):
                example = result_data['example']
                
                st.markdown(f"#### Query {i + 1}: {example['question']}")
                st.caption(f"Database: {example['db_id']} | Complexity: {result_data['result']['step2']['complexity_class'].value}")
                
                # Create comparison table
                comparison_data = []
                
                for model_name, model_results in comparison_results.items():
                    if i >= len(model_results):
                        continue
                    
                    result = model_results[i]['result']
                    
                    row = {
                        'Model': model_name,
                        'Valid': '✅' if result.get('final_is_valid', False) else '❌',
                    }
                    
                    if result.get('step10_generated'):
                        row['Exec'] = '✅' if result['step10_generated']['success'] else '❌'
                        row['Time (s)'] = f"{result['step10_generated']['execution_time']:.3f}"
                    
                    if result.get('step11'):
                        row['EX'] = '✅' if result['step11']['execution_accuracy'] else '❌'
                        row['EM'] = '✅' if result['step11']['exact_set_match'] else '❌'
                        row['Score'] = f"{result['step11']['evaluation_score']:.3f}"
                    
                    row['SQL'] = result.get('final_sql', '')[:100] + '...'
                    
                    comparison_data.append(row)
                
                if comparison_data:
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.markdown("---")


if __name__ == "__main__":
    main()