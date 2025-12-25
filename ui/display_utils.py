"""
Display Utilities for ADAPT-SQL Streamlit Application
Fixed to match new evaluation.py structure with Spider metrics
"""
import streamlit as st
import pandas as pd


def display_complexity_badge(complexity: str):
    """Display complexity with color badge"""
    if complexity == "EASY":
        st.success(f"✅ EASY")
    elif complexity == "NON_NESTED_COMPLEX":
        st.warning(f"⚠️ NON_NESTED_COMPLEX")
    else:
        st.error(f"🔴 NESTED_COMPLEX")


def display_validation_badge(is_valid: bool, validation_score: float):
    """Display validation status badge"""
    if is_valid:
        st.success(f"✅ Valid SQL (Score: {validation_score:.2f})")
    else:
        st.error(f"❌ Invalid SQL (Score: {validation_score:.2f})")


def display_schema_tab(result: dict):
    """Display schema linking results - same as before"""
    st.markdown("### Schema Linking (Three-Layer Approach)")
    
    # Overall summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Tables", len(result['step1']['schema_links']['tables']))
    with col2:
        total_cols = sum(len(cols) for cols in result['step1']['schema_links']['columns'].values())
        st.metric("Final Columns", total_cols)
    with col3:
        st.metric("Foreign Keys", len(result['step1']['schema_links']['foreign_keys']))
    
    st.markdown("---")
    
    # Check if we have layer details
    if 'layer_details' in result['step1']:
        layer_details = result['step1']['layer_details']
        
        # Create tabs for each layer
        layer_tabs = st.tabs(["🔍 Layer 1: Pre-filter", "🤖 Layer 2: LLM Analysis", "✅ Layer 3: Validation", "📊 Final Results"])
        
        # LAYER 1
        with layer_tabs[0]:
            st.markdown("#### String Matching Pre-filter")
            st.caption("Uses fuzzy matching to identify candidate tables and columns")
            
            if 'layer1' in layer_details:
                layer1 = layer_details['layer1']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Candidate Tables**")
                    st.info(f"Found {len(layer1['tables'])} candidate tables")
                    
                    if 'match_details' in layer1 and layer1['match_details']['table_matches']:
                        for match in layer1['match_details']['table_matches'][:10]:
                            reason = match['reason']
                            emoji = "🎯" if reason == "exact" else "🔍"
                            score = f" ({match['fuzzy_score']:.2f})" if reason == "fuzzy" else ""
                            st.write(f"{emoji} {match['table']}{score}")
                    else:
                        for table in sorted(layer1['tables']):
                            st.write(f"• {table}")
                
                with col2:
                    st.markdown("**Candidate Columns**")
                    total_cols = sum(len(cols) for cols in layer1['columns'].values())
                    st.info(f"Found {total_cols} candidate columns")
                    
                    if layer1['columns']:
                        count = 0
                        for table, cols in sorted(layer1['columns'].items()):
                            if cols and count < 5:
                                st.write(f"**{table}**: {', '.join(sorted(list(cols)[:3]))}{'...' if len(cols) > 3 else ''}")
                                count += 1
        
        # LAYER 2
        with layer_tabs[1]:
            st.markdown("#### LLM Analysis with Pre-filter Hints")
            st.caption("LLM analyzes question with candidate tables as hints")
            
            if 'layer2' in layer_details:
                layer2 = layer_details['layer2']
                llm_elements = layer2['parsed_elements']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**LLM Identified Tables**")
                    st.info(f"Identified {len(llm_elements['tables'])} tables")
                    
                    layer1_tables = layer_details['layer1']['tables']
                    for table in sorted(llm_elements['tables']):
                        if table in layer1_tables:
                            st.success(f"✓ {table} (from Layer 1)")
                        else:
                            st.warning(f"+ {table} (added by LLM)")
                
                with col2:
                    st.markdown("**LLM Identified Columns**")
                    total_cols = sum(len(cols) for cols in llm_elements['columns'].values())
                    st.info(f"Identified {total_cols} columns")
                    
                    for table, cols in sorted(llm_elements['columns'].items()):
                        if cols:
                            st.write(f"**{table}**: {', '.join(sorted(list(cols)[:3]))}{'...' if len(cols) > 3 else ''}")
        
        # LAYER 3
        with layer_tabs[2]:
            st.markdown("#### Post-Validation & Correction")
            st.caption("Validates and corrects LLM results against actual schema")
            
            if 'layer3' in layer_details:
                layer3 = layer_details['layer3']
                
                validation_log = layer3.get('validation_log', [])
                corrections = [v for v in validation_log if v['status'] == 'corrected']
                
                if corrections:
                    st.warning(f"**{len(corrections)} Corrections Made:**")
                    for correction in corrections:
                        if 'corrected_to' in correction:
                            st.caption(f"• {correction['element']} → {correction['corrected_to']}")
                else:
                    st.success("✓ No corrections needed - all LLM outputs were valid")
        
        # FINAL RESULTS
        with layer_tabs[3]:
            st.markdown("#### Final Schema Links")
            st.caption("Pruned schema after three-layer processing")
            
            schema_links = result['step1']['schema_links']
            
            st.markdown("**Relevant Tables:**")
            for table in sorted(schema_links['tables']):
                with st.expander(f"📊 {table}"):
                    if table in schema_links['columns']:
                        cols = schema_links['columns'][table]
                        st.write(f"**Columns ({len(cols)}):**")
                        for col in sorted(cols):
                            st.write(f"  • {col}")
    
    # Show full reasoning
    with st.expander("🔍 View Full Reasoning"):
        st.text(result['step1']['reasoning'])


def display_complexity_tab(result: dict):
    """Display complexity classification results"""
    st.markdown("### Complexity Classification")
    display_complexity_badge(result['step2']['complexity_class'].value)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• Tables: {len(result['step2']['required_tables'])}")
        st.write(f"• JOINs: {'Yes' if result['step2']['needs_joins'] else 'No'}")
        st.write(f"• Subqueries: {'Yes' if result['step2']['needs_subqueries'] else 'No'}")
    with col2:
        if result['step2']['aggregations']:
            st.write(f"• Aggregations: {', '.join(result['step2']['aggregations'])}")
        st.write(f"• GROUP BY: {'Yes' if result['step2']['has_grouping'] else 'No'}")
    
    if result['step2'].get('sub_questions'):
        st.markdown("**Sub-questions:**")
        for i, sq in enumerate(result['step2']['sub_questions'], 1):
            st.info(f"{i}. {sq}")
    
    st.markdown("**Preliminary SQL:**")
    st.code(result['step3']['predicted_sql'], language='sql')


def display_examples_tab(result: dict):
    """Display similar examples"""
    st.markdown("### Similar Examples")
    st.metric("Found", result['step4']['total_found'])
    
    for i, ex in enumerate(result['step4']['similar_examples'][:5], 1):
        score = ex.get('similarity_score', 0)
        if score >= 0.8:
            color = "HIGH"
        elif score >= 0.6:
            color = "MEDIUM"
        else:
            color = "LOW"
        
        with st.expander(f"{color} - {i}. {ex.get('question', '')[:60]}... ({score:.3f})"):
            st.markdown(f"**Question:** {ex.get('question', '')}")
            st.code(ex.get('query', ''), language='sql')


def display_routing_tab(result: dict):
    """Display routing strategy"""
    st.markdown("### Routing Strategy")
    strategy = result['step5']['strategy'].value
    st.success(f"Strategy: {strategy}")
    st.info(result['step5']['description'])


def display_sql_tab(result: dict, example: dict):
    """Display generated SQL"""
    st.markdown("### Generated SQL")
    
    # Display based on generation method
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
        
        with st.expander("NatSQL Intermediate"):
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
        
        with st.expander("NatSQL Intermediate with Sub-queries"):
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
        st.warning("No SQL generated")
    
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


def display_validation_tab(result: dict):
    """Display validation results"""
    st.markdown("### SQL Validation (Step 7)")
    
    if result.get('step7'):
        validation = result['step7']
        
        # Overall status
        col1, col2, col3 = st.columns(3)
        with col1:
            display_validation_badge(validation['is_valid'], validation['validation_score'])
        with col2:
            st.metric("Errors", len(validation['errors']))
        with col3:
            st.metric("Warnings", len(validation['warnings']))
        
        st.markdown("---")
        
        # Errors section
        if validation['errors']:
            st.markdown("### Errors")
            for i, error in enumerate(validation['errors'], 1):
                severity_label = error['severity']
                
                with st.expander(f"Error {i}: {error['type']} [{severity_label}]", expanded=True):
                    st.error(error['message'])
                    
                    if 'table' in error:
                        st.write(f"**Table:** `{error['table']}`")
                    if 'column' in error:
                        st.write(f"**Column:** `{error['column']}`")
        else:
            st.success("✅ No errors found!")
        
        st.markdown("---")
        
        # Warnings section
        if validation['warnings']:
            st.markdown("### Warnings")
            for i, warning in enumerate(validation['warnings'], 1):
                severity_label = warning['severity']
                
                with st.expander(f"Warning {i}: {warning['type']} [{severity_label}]"):
                    st.warning(warning['message'])
                    
                    if 'table' in warning:
                        st.write(f"**Table:** `{warning['table']}`")
        else:
            st.info("No warnings")
        
        st.markdown("---")
        
        # Suggestions section
        if validation['suggestions']:
            st.markdown("### Suggestions")
            for i, suggestion in enumerate(validation['suggestions'], 1):
                st.info(f"{i}. {suggestion}")
        
        # Full validation reasoning
        with st.expander("Full Validation Report"):
            st.text(validation['reasoning'])
    
    else:
        st.warning("Validation not performed")


def display_execution_tab(result: dict, example: dict):
    """Display execution results (Step 10)"""
    st.markdown("### SQL Execution (Step 10)")
    
    # Check if execution was performed
    if not result.get('step10_generated'):
        st.info("Execution not performed. Enable 'Execute SQL' option to see results.")
        return
    
    # Display generated SQL execution
    st.markdown("**Generated SQL Execution**")
    gen_exec = result['step10_generated']
    
    col1, col2 = st.columns(2)
    with col1:
        if gen_exec['success']:
            st.success("✅ Executed Successfully")
        else:
            st.error("❌ Execution Failed")
    
    with col2:
        st.metric("Execution Time", f"{gen_exec['execution_time']:.3f}s")
    
    if gen_exec['success']:
        st.markdown("**Query Results:**")
        result_df = gen_exec['result_df']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows Returned", len(result_df))
        with col2:
            st.metric("Columns", len(result_df.columns))
        
        if len(result_df) > 0:
            st.dataframe(result_df, use_container_width=True)
        else:
            st.info("Query returned no results")
    else:
        st.error(f"**Error:** {gen_exec['error_message']}")
    
    # Display ground truth execution if available
    if result.get('step10_gold'):
        st.markdown("---")
        st.markdown("**Ground Truth SQL Execution**")
        gold_exec = result['step10_gold']
        
        col1, col2 = st.columns(2)
        with col1:
            if gold_exec['success']:
                st.success("✅ Executed Successfully")
            else:
                st.error("❌ Execution Failed")
        
        with col2:
            st.metric("Execution Time", f"{gold_exec['execution_time']:.3f}s")
        
        if gold_exec['success']:
            st.markdown("**Query Results:**")
            result_df = gold_exec['result_df']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows Returned", len(result_df))
            with col2:
                st.metric("Columns", len(result_df.columns))
            
            if len(result_df) > 0:
                st.dataframe(result_df, use_container_width=True)
            else:
                st.info("Query returned no results")
        else:
            st.error(f"**Error:** {gold_exec['error_message']}")


def display_evaluation_tab(result: dict, example: dict):
    """Display evaluation results (Step 11) - FIXED for new Spider metrics"""
    st.markdown("### Evaluation (Step 11)")
    
    # Check if evaluation was performed
    if not result.get('step11'):
        st.info("Evaluation not performed. Both execution and ground truth SQL are required.")
        return
    
    eval_result = result['step11']
    
    # Overall score
    st.markdown("**Overall Evaluation Score**")
    
    score = eval_result['evaluation_score']
    
    # Display score with progress bar
    col1, col2 = st.columns([1, 3])
    with col1:
        if score >= 0.9:
            st.success(f"**{score:.2f}**")
            st.caption("Grade: A (Excellent)")
        elif score >= 0.7:
            st.info(f"**{score:.2f}**")
            st.caption("Grade: B (Good)")
        elif score >= 0.5:
            st.warning(f"**{score:.2f}**")
            st.caption("Grade: C (Fair)")
        else:
            st.error(f"**{score:.2f}**")
            st.caption("Grade: D (Needs Improvement)")
    
    with col2:
        st.progress(score)
    
    st.markdown("---")
    
    # Spider Benchmark Metrics
    st.markdown("**Spider Benchmark Metrics**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Primary Metric: Execution Accuracy (EX)**")
        if eval_result['execution_accuracy']:
            st.success("✅ PASS - Results match exactly")
            st.caption("Most important metric - queries produce identical results")
        else:
            st.error("❌ FAIL - Results don't match")
            st.caption("Critical issue - output differs from ground truth")
    
    with col2:
        st.markdown("**Secondary Metric: Exact-Set-Match (EM)**")
        if eval_result['exact_set_match']:
            st.success("✅ PASS - SQL structure matches")
            st.caption("SQL clauses match when treated as sets")
        else:
            st.error("❌ FAIL - SQL structure differs")
            st.caption("Different structure (acceptable if EX passes)")
    
    st.markdown("---")
    
    # Component-level breakdown
    st.markdown("**Component-Level Match (for Exact-Set-Match)**")
    
    component_match = eval_result.get('component_match', {})
    
    if component_match:
        cols = st.columns(min(len(component_match), 4))
        
        for i, (component, matches) in enumerate(sorted(component_match.items())):
            col_idx = i % 4
            with cols[col_idx]:
                if matches:
                    st.success(f"✅ {component}")
                else:
                    st.error(f"❌ {component}")
    
    st.markdown("---")
    
    # Explanation of metrics
    with st.expander("ℹ️ Understanding Spider Metrics"):
        st.markdown("""
        **Execution Accuracy (EX)** - Primary Metric (80% weight)
        - Compares actual query results between generated and ground truth SQL
        - This is the **most important** metric - queries must produce identical results
        - Multiple different SQL queries can be correct if they produce the same results
        
        **Exact-Set-Match (EM)** - Secondary Metric (20% weight)
        - Treats each SQL clause (SELECT, FROM, WHERE, etc.) as a set
        - Compares whether clauses match structurally
        - Less important than EX - different structures can be equally correct
        
        **Component Match** - Detailed Breakdown
        - Shows which specific SQL components match or differ
        - Helps understand where the SQL structure differs
        - Used for debugging and improvement suggestions
        
        **Composite Score**: (EX × 0.80) + (EM × 0.20)
        - Weighted heavily toward execution accuracy
        - Reflects research-backed evaluation priorities
        """)
    
    # Full evaluation reasoning
    with st.expander("📋 Full Evaluation Report"):
        st.text(eval_result['reasoning'])


def display_retry_history_tab(retry_result: dict):
    """Display retry attempt history - FIXED for new evaluation structure"""
    st.markdown("### Retry History")
    
    total_attempts = retry_result['total_attempts']
    success = retry_result['success']
    
    st.markdown(f"**Total Attempts:** {total_attempts}")
    st.markdown(f"**Final Status:** {'✅ SUCCESS' if success else '⚠️ MAX RETRIES REACHED'}")
    
    st.markdown("---")
    
    # Display each attempt
    for i, attempt_info in enumerate(retry_result['attempt_history']):
        result = attempt_info['result']
        attempt_num = attempt_info['attempt_number']
        
        # Determine status based on new evaluation structure
        has_eval = result.get('step11') is not None
        if has_eval:
            exec_acc = result['step11'].get('execution_accuracy', False)
            eval_score = result['step11'].get('evaluation_score', 0)
            status = "✅ Success" if exec_acc else "⚠️ Issues"
        else:
            status = "⚠️ No Evaluation"
        
        with st.expander(
            f"Attempt {attempt_num} - {status}",
            expanded=(i == len(retry_result['attempt_history']) - 1)
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Execution**")
                if result.get('step10_generated'):
                    if result['step10_generated']['success']:
                        st.success("✅ Success")
                    else:
                        st.error("❌ Failed")
                        st.caption(result['step10_generated']['error_message'][:50])
            
            with col2:
                st.markdown("**Validation**")
                if result.get('step7'):
                    val_score = result['step7']['validation_score']
                    if result['step7']['is_valid']:
                        st.success(f"✅ Valid ({val_score:.2f})")
                    else:
                        st.error(f"❌ Invalid ({val_score:.2f})")
            
            with col3:
                st.markdown("**Evaluation**")
                if has_eval:
                    # Use new evaluation structure
                    exec_acc = result['step11'].get('execution_accuracy', False)
                    eval_score = result['step11'].get('evaluation_score', 0)
                    
                    if exec_acc:
                        st.success(f"✅ EX: 1.0")
                        st.caption(f"Score: {eval_score:.2f}")
                    else:
                        st.error(f"❌ EX: 0.0")
                        st.caption(f"Score: {eval_score:.2f}")
                else:
                    st.info("No evaluation")
            
            st.markdown("**Generated SQL:**")
            st.code(result.get('final_sql', 'N/A'), language='sql')
            
            # Show metrics details if available
            if has_eval:
                st.markdown("**Spider Metrics:**")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    exec_acc = result['step11'].get('execution_accuracy', False)
                    st.write(f"• Execution Accuracy (EX): {'✅ 1.0' if exec_acc else '❌ 0.0'}")
                
                with metrics_col2:
                    exact_set = result['step11'].get('exact_set_match', False)
                    st.write(f"• Exact-Set-Match (EM): {'✅ 1.0' if exact_set else '❌ 0.0'}")
    
    # Show reasoning
    with st.expander("📋 Retry Process Reasoning"):
        st.text(retry_result['reasoning'])