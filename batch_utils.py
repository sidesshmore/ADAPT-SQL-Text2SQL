"""
Batch Processing Utilities for ADAPT-SQL
Helper functions for batch processing and display
"""
import streamlit as st
import pandas as pd
import json
from typing import Dict, List


def display_batch_summary(results: List[Dict]):
    """Display summary statistics for batch results"""
    st.markdown("### Summary Statistics")
    
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


def display_complexity_distribution(results: List[Dict]):
    """Display complexity distribution"""
    st.markdown("### Complexity Distribution")
    
    complexity_counts = {}
    for r in results:
        complexity = r['result']['step2']['complexity_class'].value
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("EASY", complexity_counts.get("EASY", 0))
    with col2:
        st.metric("NON_NESTED_COMPLEX", complexity_counts.get("NON_NESTED_COMPLEX", 0))
    with col3:
        st.metric("NESTED_COMPLEX", complexity_counts.get("NESTED_COMPLEX", 0))


def display_execution_summary(results: List[Dict]):
    """Display execution statistics"""
    st.markdown("### Execution Statistics")
    
    executed_count = sum(1 for r in results if r['result'].get('step10_generated'))
    if executed_count == 0:
        st.info("No queries were executed")
        return
    
    success_count = sum(
        1 for r in results 
        if r['result'].get('step10_generated', {}).get('success', False)
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Executed", executed_count)
    with col2:
        st.metric("Successful", success_count)
    with col3:
        success_rate = (success_count / executed_count * 100) if executed_count > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")


def display_evaluation_summary(results: List[Dict]):
    """Display evaluation statistics"""
    st.markdown("### Evaluation Statistics")
    
    evaluated_count = sum(1 for r in results if r['result'].get('step11'))
    if evaluated_count == 0:
        st.info("No queries were evaluated")
        return
    
    exec_accuracy_count = sum(
        1 for r in results 
        if r['result'].get('step11', {}).get('execution_accuracy', False)
    )
    
    avg_eval_score = sum(
        r['result'].get('step11', {}).get('evaluation_score', 0) 
        for r in results if r['result'].get('step11')
    ) / evaluated_count if evaluated_count > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Evaluated", evaluated_count)
    with col2:
        st.metric("Execution Accuracy", exec_accuracy_count)
    with col3:
        st.metric("Avg Score", f"{avg_eval_score:.2f}")


def display_query_summary_card(idx: int, example: Dict, result: Dict):
    """Display a summary card for a single query"""
    # Get complexity
    complexity = result['step2']['complexity_class'].value
    complexity_color = {
        "EASY": "success",
        "NON_NESTED_COMPLEX": "warning",
        "NESTED_COMPLEX": "error"
    }.get(complexity, "info")
    
    # Get validation status
    is_valid = result.get('step7', {}).get('is_valid', False)
    validation_score = result.get('step7', {}).get('validation_score', 0)
    
    # Get execution status
    exec_success = result.get('step10_generated', {}).get('success', None)
    
    # Get evaluation score
    eval_score = result.get('step11', {}).get('evaluation_score', None)
    
    # Display card
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.markdown(f"**#{idx + 1}:** {example['question'][:60]}...")
        st.caption(f"Database: {example['db_id']}")
    
    with col2:
        if complexity == "EASY":
            st.success(complexity)
        elif complexity == "NON_NESTED_COMPLEX":
            st.warning("NON_NESTED")
        else:
            st.error("NESTED")
    
    with col3:
        if is_valid:
            st.success(f"Valid: {validation_score:.2f}")
        else:
            st.error(f"Invalid: {validation_score:.2f}")
    
    with col4:
        if eval_score is not None:
            if eval_score >= 0.7:
                st.success(f"Score: {eval_score:.2f}")
            elif eval_score >= 0.5:
                st.warning(f"Score: {eval_score:.2f}")
            else:
                st.error(f"Score: {eval_score:.2f}")
        elif exec_success is not None:
            if exec_success:
                st.info("Exec: OK")
            else:
                st.error("Exec: FAIL")
        else:
            st.info("Not executed")


def display_query_details(idx: int, example: Dict, result: Dict):
    """Display detailed results for a single query"""
    from display_utils import (
        display_schema_tab,
        display_complexity_tab,
        display_examples_tab,
        display_routing_tab,
        display_sql_tab,
        display_validation_tab,
        display_execution_tab,
        display_evaluation_tab
    )
    
    with st.expander(f"Query #{idx + 1}: {example['question'][:80]}...", expanded=False):
        st.markdown(f"**Database:** `{example['db_id']}`")
        st.markdown(f"**Question:** {example['question']}")
        
        # Create tabs
        tabs = st.tabs([
            "Schema", "Complexity", "Examples", "Route", 
            "SQL", "Validation", "Execution", "Evaluation"
        ])
        
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
            if r['result'].get('step7', {}).get('is_valid', False)
        ]
    elif filter_validity == "Invalid Only":
        filtered = [
            r for r in filtered 
            if not r['result'].get('step7', {}).get('is_valid', True)
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
    if filter_evaluation == "High Score (>=0.7)":
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
        generated_sql = None
        if r['result'].get('step6a'):
            generated_sql = r['result']['step6a']['generated_sql']
        elif r['result'].get('step6b'):
            generated_sql = r['result']['step6b']['generated_sql']
        elif r['result'].get('step6c'):
            generated_sql = r['result']['step6c']['generated_sql']
        
        # Get execution info
        exec_success = None
        exec_time = None
        if r['result'].get('step10_generated'):
            exec_success = r['result']['step10_generated']['success']
            exec_time = r['result']['step10_generated']['execution_time']
        
        # Get evaluation info
        eval_score = None
        exec_accuracy = None
        if r['result'].get('step11'):
            eval_score = r['result']['step11']['evaluation_score']
            exec_accuracy = r['result']['step11']['execution_accuracy']
        
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
            'Num_Warnings': len(r['result'].get('step7', {}).get('warnings', [])),
            'Execution_Success': exec_success,
            'Execution_Time': exec_time,
            'Evaluation_Score': eval_score,
            'Execution_Accuracy': exec_accuracy
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
            'result': result_copy
        })
    
    return json.dumps(export_data, indent=2)


def display_error_analysis(results: List[Dict]):
    """Display analysis of common errors"""
    st.markdown("### Error Analysis")
    
    # Collect all errors
    error_types = {}
    for r in results:
        errors = r['result'].get('step7', {}).get('errors', [])
        for error in errors:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    if not error_types:
        st.success("No validation errors found!")
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