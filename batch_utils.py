"""
Batch Processing Utilities for ADAPT-SQL
Updated with full UI display matching app.py
"""
import streamlit as st
import pandas as pd
import json
from typing import Dict, List


def display_batch_summary(results: List[Dict]):
    """Display summary statistics for batch results"""
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Queries", len(results))
    
    with col2:
        valid_count = sum(1 for r in results if r['result'].get('final_is_valid', False))
        st.metric("Valid SQL", valid_count)
    
    with col3:
        if any(r['result'].get('step10_generated') for r in results):
            exec_success = sum(1 for r in results if r['result'].get('step10_generated', {}).get('success', False))
            st.metric("Executed", exec_success)
        else:
            st.metric("Executed", "N/A")
    
    with col4:
        if any(r['result'].get('step11') for r in results):
            ex_acc = sum(1 for r in results if r['result'].get('step11', {}).get('execution_accuracy', False))
            st.metric("EX = 1.0", ex_acc)
        else:
            st.metric("EX = 1.0", "N/A")
    
    with col5:
        if any(r['result'].get('step11') for r in results):
            avg_score = sum(r['result'].get('step11', {}).get('evaluation_score', 0) for r in results if r['result'].get('step11')) / len([r for r in results if r['result'].get('step11')])
            st.metric("Avg Score", f"{avg_score:.2f}")
        else:
            st.metric("Avg Score", "N/A")


def display_complexity_distribution(results: List[Dict]):
    """Display complexity distribution"""
    st.markdown("### 📈 Complexity Distribution")
    
    complexity_counts = {}
    for r in results:
        complexity = r['result']['step2']['complexity_class'].value
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        count = complexity_counts.get("EASY", 0)
        pct = (count / len(results) * 100) if results else 0
        st.metric("EASY", count, f"{pct:.1f}%")
    with col2:
        count = complexity_counts.get("NON_NESTED_COMPLEX", 0)
        pct = (count / len(results) * 100) if results else 0
        st.metric("NON_NESTED", count, f"{pct:.1f}%")
    with col3:
        count = complexity_counts.get("NESTED_COMPLEX", 0)
        pct = (count / len(results) * 100) if results else 0
        st.metric("NESTED", count, f"{pct:.1f}%")


def display_execution_summary(results: List[Dict]):
    """Display execution statistics"""
    st.markdown("### ⚡ Execution Statistics")
    
    executed_count = sum(1 for r in results if r['result'].get('step10_generated'))
    if executed_count == 0:
        st.info("No queries were executed")
        return
    
    success_count = sum(
        1 for r in results 
        if r['result'].get('step10_generated', {}).get('success', False)
    )
    
    avg_time = sum(
        r['result'].get('step10_generated', {}).get('execution_time', 0)
        for r in results if r['result'].get('step10_generated', {}).get('success')
    ) / success_count if success_count > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Executed", executed_count)
    with col2:
        st.metric("Successful", success_count)
    with col3:
        success_rate = (success_count / executed_count * 100) if executed_count > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        st.metric("Avg Exec Time", f"{avg_time:.3f}s")


def display_evaluation_summary(results: List[Dict]):
    """Display evaluation statistics"""
    st.markdown("### 🎯 Evaluation Statistics (Spider Metrics)")
    
    evaluated_count = sum(1 for r in results if r['result'].get('step11'))
    if evaluated_count == 0:
        st.info("No queries were evaluated")
        return
    
    exec_accuracy_count = sum(
        1 for r in results 
        if r['result'].get('step11', {}).get('execution_accuracy', False)
    )
    
    exact_set_count = sum(
        1 for r in results 
        if r['result'].get('step11', {}).get('exact_set_match', False)
    )
    
    avg_eval_score = sum(
        r['result'].get('step11', {}).get('evaluation_score', 0) 
        for r in results if r['result'].get('step11')
    ) / evaluated_count if evaluated_count > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Evaluated", evaluated_count)
    with col2:
        ex_rate = (exec_accuracy_count / evaluated_count * 100) if evaluated_count > 0 else 0
        st.metric("EX = 1.0", exec_accuracy_count, f"{ex_rate:.1f}%")
    with col3:
        em_rate = (exact_set_count / evaluated_count * 100) if evaluated_count > 0 else 0
        st.metric("EM = 1.0", exact_set_count, f"{em_rate:.1f}%")
    with col4:
        st.metric("Avg Score", f"{avg_eval_score:.2f}")


def display_retry_summary(results: List[Dict]):
    """Display retry statistics"""
    st.markdown("### 🔄 Retry Statistics")
    
    retry_count = sum(1 for r in results if r.get('retry_result'))
    if retry_count == 0:
        st.info("No queries used full pipeline retry")
        return
    
    total_attempts = sum(
        r['retry_result']['total_attempts'] 
        for r in results if r.get('retry_result')
    )
    
    success_count = sum(
        1 for r in results 
        if r.get('retry_result', {}).get('success', False)
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Queries with Retry", retry_count)
    with col2:
        avg_attempts = total_attempts / retry_count if retry_count > 0 else 0
        st.metric("Avg Attempts", f"{avg_attempts:.1f}")
    with col3:
        st.metric("Retry Success", success_count)
    with col4:
        success_rate = (success_count / retry_count * 100) if retry_count > 0 else 0
        st.metric("Retry Success Rate", f"{success_rate:.1f}%")


def display_query_summary_card(idx: int, example: Dict, result: Dict, retry_result: Dict = None):
    """Display a summary card for a single query"""
    # Get complexity
    complexity = result['step2']['complexity_class'].value
    
    # Get validation status
    is_valid = result.get('final_is_valid', False)
    validation_score = result.get('step7', {}).get('validation_score', 0)
    
    # Get execution status
    exec_success = result.get('step10_generated', {}).get('success', None)
    
    # Get evaluation score
    eval_score = result.get('step11', {}).get('evaluation_score', None)
    ex_acc = result.get('step11', {}).get('execution_accuracy', None)
    
    # Display card
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"**#{idx + 1}:** {example['question'][:80]}...")
            st.caption(f"📁 Database: {example['db_id']}")
            if retry_result:
                st.caption(f"🔄 Retry: {retry_result['total_attempts']} attempts | {'✅ Success' if retry_result['success'] else '⚠️ Max reached'}")
        
        with col2:
            if complexity == "EASY":
                st.success("✅ EASY")
            elif complexity == "NON_NESTED_COMPLEX":
                st.warning("⚠️ NON_NESTED")
            else:
                st.error("🔴 NESTED")
        
        with col3:
            if is_valid:
                st.success(f"✅ {validation_score:.2f}")
            else:
                st.error(f"❌ {validation_score:.2f}")
        
        with col4:
            if exec_success is not None:
                if exec_success:
                    st.success("✅ Exec")
                else:
                    st.error("❌ Exec")
            else:
                st.info("⊘ No Exec")
        
        with col5:
            if eval_score is not None:
                if ex_acc:
                    st.success(f"✅ {eval_score:.2f}")
                elif eval_score >= 0.5:
                    st.warning(f"⚠️ {eval_score:.2f}")
                else:
                    st.error(f"❌ {eval_score:.2f}")
            else:
                st.info("⊘ No Eval")


def display_query_details(idx: int, example: Dict, result: Dict, retry_result: Dict = None):
    """Display detailed results for a single query with full UI matching app.py"""
    from display_utils import (
        display_schema_tab,
        display_complexity_tab,
        display_examples_tab,
        display_routing_tab,
        display_sql_tab,
        display_validation_tab,
        display_execution_tab,
        display_evaluation_tab,
        display_retry_history_tab
    )
    
    with st.expander(f"📋 Query #{idx + 1}: {example['question'][:100]}...", expanded=False):
        # Header with key info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Question:** {example['question']}")
        with col2:
            st.markdown(f"**Database:** `{example['db_id']}`")
        with col3:
            complexity = result['step2']['complexity_class'].value
            if complexity == "EASY":
                st.success(f"✅ {complexity}")
            elif complexity == "NON_NESTED_COMPLEX":
                st.warning(f"⚠️ {complexity}")
            else:
                st.error(f"🔴 {complexity}")
        
        st.markdown("---")
        
        # Create tabs - add Retry History if available
        if retry_result:
            tabs = st.tabs([
                "📊 Schema", "🎯 Complexity", "🔍 Examples", "🔀 Route", 
                "💻 SQL", "✅ Validation", "⚡ Execution", "📈 Evaluation", "🔄 Retry History"
            ])
            has_retry_tab = True
        else:
            tabs = st.tabs([
                "📊 Schema", "🎯 Complexity", "🔍 Examples", "🔀 Route", 
                "💻 SQL", "✅ Validation", "⚡ Execution", "📈 Evaluation"
            ])
            has_retry_tab = False
        
        with tabs[0]:
            display_schema_tab(result)
        
        with tabs[1]:
            display_complexity_tab(result)
        
        with tabs[2]:
            display_examples_tab(result)
        
        with tabs[3]:
            display_routing_tab(result)
        
        with tabs[4]:
            display_sql_tab(result, example)
        
        with tabs[5]:
            display_validation_tab(result)
        
        with tabs[6]:
            display_execution_tab(result, example)
        
        with tabs[7]:
            display_evaluation_tab(result, example)
        
        if has_retry_tab:
            with tabs[8]:
                display_retry_history_tab(retry_result)


def filter_results(results: List[Dict], filter_complexity: List[str], filter_validity: str, 
                   filter_execution: str = "All", filter_evaluation: str = "All") -> List[Dict]:
    """Filter results based on criteria"""
    filtered = results
    
    # Filter by complexity
    if filter_complexity:
        filtered = [
            r for r in filtered 
            if r['result']['step2']['complexity_class'].value in filter_complexity
        ]
    
    # Filter by validity
    if filter_validity == "Valid Only":
        filtered = [
            r for r in filtered 
            if r['result'].get('final_is_valid', False)
        ]
    elif filter_validity == "Invalid Only":
        filtered = [
            r for r in filtered 
            if not r['result'].get('final_is_valid', True)
        ]
    
    # Filter by execution
    if filter_execution == "Success Only":
        filtered = [
            r for r in filtered
            if r['result'].get('step10_generated', {}).get('success', False)
        ]
    elif filter_execution == "Failed Only":
        filtered = [
            r for r in filtered
            if r['result'].get('step10_generated') and 
               not r['result']['step10_generated'].get('success', True)
        ]
    
    # Filter by evaluation
    if filter_evaluation == "EX = 1.0 Only":
        filtered = [
            r for r in filtered
            if r['result'].get('step11', {}).get('execution_accuracy', False)
        ]
    elif filter_evaluation == "EX = 0.0 Only":
        filtered = [
            r for r in filtered
            if r['result'].get('step11') and 
               not r['result']['step11'].get('execution_accuracy', True)
        ]
    elif filter_evaluation == "High Score (>=0.7)":
        filtered = [
            r for r in filtered
            if r['result'].get('step11', {}).get('evaluation_score', 0) >= 0.7
        ]
    elif filter_evaluation == "Low Score (<0.5)":
        filtered = [
            r for r in filtered
            if r['result'].get('step11', {}).get('evaluation_score', 1) < 0.5
        ]
    
    return filtered


def export_summary_csv(results: List[Dict]) -> str:
    """Create CSV export of summary data"""
    summary_data = []
    
    for r in results:
        # Get generated SQL
        generated_sql = r['result'].get('final_sql', '')
        
        # Get execution info
        exec_success = None
        exec_time = None
        if r['result'].get('step10_generated'):
            exec_success = r['result']['step10_generated']['success']
            exec_time = r['result']['step10_generated']['execution_time']
        
        # Get evaluation info
        eval_score = None
        exec_accuracy = None
        exact_set_match = None
        if r['result'].get('step11'):
            eval_score = r['result']['step11']['evaluation_score']
            exec_accuracy = r['result']['step11']['execution_accuracy']
            exact_set_match = r['result']['step11']['exact_set_match']
        
        # Get retry info
        retry_attempts = None
        retry_success = None
        if r.get('retry_result'):
            retry_attempts = r['retry_result']['total_attempts']
            retry_success = r['retry_result']['success']
        
        summary_data.append({
            'Index': r['index'],
            'Database': r['example']['db_id'],
            'Question': r['example']['question'],
            'Complexity': r['result']['step2']['complexity_class'].value,
            'Strategy': r['result']['step5']['strategy'].value,
            'Generated_SQL': generated_sql,
            'Ground_Truth_SQL': r['example'].get('query', ''),
            'Is_Valid': r['result'].get('final_is_valid', None),
            'Validation_Score': r['result'].get('step7', {}).get('validation_score', None),
            'Num_Errors': len(r['result'].get('step7', {}).get('errors', [])),
            'Num_Warnings': len(r['result'].get('step7', {}).get('warnings', [])),
            'Execution_Success': exec_success,
            'Execution_Time': exec_time,
            'Evaluation_Score': eval_score,
            'Execution_Accuracy_EX': exec_accuracy,
            'Exact_Set_Match_EM': exact_set_match,
            'Retry_Attempts': retry_attempts,
            'Retry_Success': retry_success
        })
    
    df = pd.DataFrame(summary_data)
    return df.to_csv(index=False)


def export_full_json(results: List[Dict]) -> str:
    """Create JSON export of full results"""
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
            elif key in ['step10_generated', 'step10_gold']:
                # Handle DataFrames
                if value and 'result_df' in value:
                    value_copy = value.copy()
                    if value['result_df'] is not None:
                        value_copy['result_df'] = value['result_df'].to_dict('records')
                    result_copy[key] = value_copy
                else:
                    result_copy[key] = value
            else:
                result_copy[key] = value
        
        export_data.append({
            'index': r['index'],
            'example': r['example'],
            'result': result_copy,
            'retry_result': r.get('retry_result')
        })
    
    return json.dumps(export_data, indent=2)


def display_error_analysis(results: List[Dict]):
    """Display analysis of common errors"""
    st.markdown("### 🔍 Error Analysis")
    
    # Collect all errors
    error_types = {}
    for r in results:
        errors = r['result'].get('step7', {}).get('errors', [])
        for error in errors:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    if not error_types:
        st.success("✅ No validation errors found!")
        return
    
    # Display error distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Error Type Distribution:**")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- {error_type}: {count}")
    
    with col2:
        st.markdown("**Most Common Errors:**")
        top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (error_type, count) in enumerate(top_errors, 1):
            st.write(f"{i}. {error_type} ({count} occurrences)")