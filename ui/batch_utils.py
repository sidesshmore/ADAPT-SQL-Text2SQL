"""
Streamlined Batch Processing Utilities - Focus on Statistics
"""
import streamlit as st
import pandas as pd
import pickle
from typing import Dict, List
from pathlib import Path
from datetime import datetime


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(results: List[Dict], timestamp: str, output_dir: Path, final: bool = False) -> Path:
    """Save checkpoint to disk"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if final:
        checkpoint_name = f"final_checkpoint_{timestamp.replace(' ', '_').replace(':', '-')}.pkl"
    else:
        checkpoint_name = f"checkpoint_{len(results)}_{timestamp.replace(' ', '_').replace(':', '-')}.pkl"
    
    checkpoint_path = output_dir / checkpoint_name
    
    checkpoint_data = {
        'results': results,
        'timestamp': timestamp,
        'num_results': len(results),
        'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'is_final': final
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint from disk"""
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None


def get_checkpoint_files(output_dir: Path) -> List[Path]:
    """Get list of checkpoint files"""
    if not output_dir.exists():
        return []
    
    checkpoint_files = list(output_dir.glob("checkpoint_*.pkl")) + list(output_dir.glob("final_checkpoint_*.pkl"))
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return checkpoint_files


# ============================================================================
# COMPREHENSIVE STATISTICS DISPLAY
# ============================================================================

def display_comprehensive_statistics(results: List[Dict]):
    """Display comprehensive at-a-glance statistics"""
    
    st.markdown("## 📊 System Performance Overview")
    st.markdown("---")
    
    # =====================================================================
    # TOP-LEVEL METRICS
    # =====================================================================
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total = len(results)
    
    with col1:
        st.metric("Total Queries", total)
    
    with col2:
        valid_count = sum(1 for r in results if r['result'].get('final_is_valid', False))
        valid_rate = (valid_count / total * 100) if total > 0 else 0
        st.metric("Valid SQL", f"{valid_rate:.1f}%", f"{valid_count}/{total}")
    
    with col3:
        if any(r['result'].get('step10_generated') for r in results):
            exec_success = sum(1 for r in results if r['result'].get('step10_generated', {}).get('success', False))
            exec_rate = (exec_success / total * 100) if total > 0 else 0
            st.metric("Execution", f"{exec_rate:.1f}%", f"{exec_success}/{total}")
        else:
            st.metric("Execution", "N/A")
    
    with col4:
        if any(r['result'].get('step11') for r in results):
            ex_acc = sum(1 for r in results if r['result'].get('step11', {}).get('execution_accuracy', False))
            ex_rate = (ex_acc / total * 100) if total > 0 else 0
            st.metric("EX = 1.0", f"{ex_rate:.1f}%", f"{ex_acc}/{total}")
        else:
            st.metric("EX = 1.0", "N/A")
    
    with col5:
        if any(r['result'].get('step11') for r in results):
            em_match = sum(1 for r in results if r['result'].get('step11', {}).get('exact_set_match', False))
            em_rate = (em_match / total * 100) if total > 0 else 0
            st.metric("EM = 1.0", f"{em_rate:.1f}%", f"{em_match}/{total}")
        else:
            st.metric("EM = 1.0", "N/A")
    
    with col6:
        if any(r['result'].get('step11') for r in results):
            avg_score = sum(r['result'].get('step11', {}).get('evaluation_score', 0) 
                          for r in results if r['result'].get('step11')) / len([r for r in results if r['result'].get('step11')])
            st.metric("Avg Score", f"{avg_score:.3f}")
        else:
            st.metric("Avg Score", "N/A")
    
    st.markdown("---")
    
    # =====================================================================
    # COMPLEXITY BREAKDOWN
    # =====================================================================
    st.markdown("### 📈 Complexity Distribution & Performance")
    
    complexity_stats = {}
    for complexity in ["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"]:
        complexity_results = [r for r in results if r['result']['step2']['complexity_class'].value == complexity]
        
        if not complexity_results:
            continue
        
        count = len(complexity_results)
        valid = sum(1 for r in complexity_results if r['result'].get('final_is_valid', False))
        
        stats = {
            'count': count,
            'pct': (count / total * 100) if total > 0 else 0,
            'valid': valid,
            'valid_rate': (valid / count * 100) if count > 0 else 0
        }
        
        if any(r['result'].get('step11') for r in complexity_results):
            ex_acc = sum(1 for r in complexity_results if r['result'].get('step11', {}).get('execution_accuracy', False))
            eval_count = len([r for r in complexity_results if r['result'].get('step11')])
            avg_score = sum(r['result'].get('step11', {}).get('evaluation_score', 0) 
                          for r in complexity_results if r['result'].get('step11')) / eval_count if eval_count > 0 else 0
            
            stats['ex_acc'] = ex_acc
            stats['ex_rate'] = (ex_acc / eval_count * 100) if eval_count > 0 else 0
            stats['avg_score'] = avg_score
        
        complexity_stats[complexity] = stats
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "EASY" in complexity_stats:
            stats = complexity_stats["EASY"]
            st.success("**✅ EASY**")
            st.write(f"**Count:** {stats['count']} ({stats['pct']:.1f}%)")
            st.write(f"**Valid:** {stats['valid']}/{stats['count']} ({stats['valid_rate']:.1f}%)")
            if 'ex_acc' in stats:
                st.write(f"**EX=1.0:** {stats['ex_acc']} ({stats['ex_rate']:.1f}%)")
                st.write(f"**Avg Score:** {stats['avg_score']:.3f}")
        else:
            st.info("No EASY queries")
    
    with col2:
        if "NON_NESTED_COMPLEX" in complexity_stats:
            stats = complexity_stats["NON_NESTED_COMPLEX"]
            st.warning("**⚠️ NON_NESTED**")
            st.write(f"**Count:** {stats['count']} ({stats['pct']:.1f}%)")
            st.write(f"**Valid:** {stats['valid']}/{stats['count']} ({stats['valid_rate']:.1f}%)")
            if 'ex_acc' in stats:
                st.write(f"**EX=1.0:** {stats['ex_acc']} ({stats['ex_rate']:.1f}%)")
                st.write(f"**Avg Score:** {stats['avg_score']:.3f}")
        else:
            st.info("No NON_NESTED queries")
    
    with col3:
        if "NESTED_COMPLEX" in complexity_stats:
            stats = complexity_stats["NESTED_COMPLEX"]
            st.error("**🔴 NESTED**")
            st.write(f"**Count:** {stats['count']} ({stats['pct']:.1f}%)")
            st.write(f"**Valid:** {stats['valid']}/{stats['count']} ({stats['valid_rate']:.1f}%)")
            if 'ex_acc' in stats:
                st.write(f"**EX=1.0:** {stats['ex_acc']} ({stats['ex_rate']:.1f}%)")
                st.write(f"**Avg Score:** {stats['avg_score']:.3f}")
        else:
            st.info("No NESTED queries")
    
    st.markdown("---")
    
    # =====================================================================
    # EXECUTION STATISTICS
    # =====================================================================
    if any(r['result'].get('step10_generated') for r in results):
        st.markdown("### ⚡ Execution Performance")
        
        executed_results = [r for r in results if r['result'].get('step10_generated')]
        success_results = [r for r in executed_results if r['result']['step10_generated']['success']]
        failed_results = [r for r in executed_results if not r['result']['step10_generated']['success']]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Executed", len(executed_results))
        
        with col2:
            success_rate = (len(success_results) / len(executed_results) * 100) if executed_results else 0
            st.metric("Success Rate", f"{success_rate:.1f}%", f"{len(success_results)}/{len(executed_results)}")
        
        with col3:
            if success_results:
                avg_time = sum(r['result']['step10_generated']['execution_time'] for r in success_results) / len(success_results)
                st.metric("Avg Time", f"{avg_time:.3f}s")
            else:
                st.metric("Avg Time", "N/A")
        
        with col4:
            if failed_results:
                st.metric("Failed", len(failed_results), delta=f"-{len(failed_results)}", delta_color="inverse")
            else:
                st.metric("Failed", 0, delta="✅")
        
        # Common execution errors
        if failed_results:
            st.markdown("**Common Execution Errors:**")
            error_types = {}
            for r in failed_results:
                error = r['result']['step10_generated']['error_message']
                error_type = error.split(':')[0] if ':' in error else error[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.caption(f"• {error_type}: {count} occurrences")
        
        st.markdown("---")
    
    # =====================================================================
    # SPIDER BENCHMARK METRICS
    # =====================================================================
    if any(r['result'].get('step11') for r in results):
        st.markdown("### 🎯 Spider Benchmark Results")
        
        eval_results = [r for r in results if r['result'].get('step11')]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🥇 Execution Accuracy (EX)**")
            st.caption("Primary metric - Results match exactly")
            
            ex_perfect = sum(1 for r in eval_results if r['result']['step11']['execution_accuracy'])
            ex_failed = len(eval_results) - ex_perfect
            ex_rate = (ex_perfect / len(eval_results) * 100) if eval_results else 0
            
            st.metric("EX = 1.0", f"{ex_rate:.1f}%", f"{ex_perfect}/{len(eval_results)}")
            st.progress(ex_rate / 100)
            
            if ex_failed > 0:
                st.caption(f"❌ Failed: {ex_failed} queries")
        
        with col2:
            st.markdown("**🥈 Exact-Set-Match (EM)**")
            st.caption("Secondary metric - SQL structure matches")
            
            em_perfect = sum(1 for r in eval_results if r['result']['step11']['exact_set_match'])
            em_failed = len(eval_results) - em_perfect
            em_rate = (em_perfect / len(eval_results) * 100) if eval_results else 0
            
            st.metric("EM = 1.0", f"{em_rate:.1f}%", f"{em_perfect}/{len(eval_results)}")
            st.progress(em_rate / 100)
            
            if em_failed > 0:
                st.caption(f"❌ Failed: {em_failed} queries")
        
        with col3:
            st.markdown("**🏆 Composite Score**")
            st.caption("Weighted: 80% EX + 20% EM")
            
            avg_score = sum(r['result']['step11']['evaluation_score'] for r in eval_results) / len(eval_results)
            
            st.metric("Average", f"{avg_score:.3f}")
            st.progress(avg_score)
            
            # Score distribution
            perfect = sum(1 for r in eval_results if r['result']['step11']['evaluation_score'] >= 0.95)
            good = sum(1 for r in eval_results if 0.7 <= r['result']['step11']['evaluation_score'] < 0.95)
            fair = sum(1 for r in eval_results if 0.5 <= r['result']['step11']['evaluation_score'] < 0.7)
            poor = sum(1 for r in eval_results if r['result']['step11']['evaluation_score'] < 0.5)
            
            st.caption(f"✅ Perfect (≥0.95): {perfect}")
            st.caption(f"🟢 Good (0.7-0.95): {good}")
            st.caption(f"🟡 Fair (0.5-0.7): {fair}")
            st.caption(f"🔴 Poor (<0.5): {poor}")
        
        st.markdown("---")
    
    # =====================================================================
    # RETRY STATISTICS
    # =====================================================================
    if any(r.get('retry_result') for r in results):
        st.markdown("### 🔄 Full Pipeline Retry Performance")
        
        retry_results = [r for r in results if r.get('retry_result')]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Queries with Retry", len(retry_results))
        
        with col2:
            total_attempts = sum(r['retry_result']['total_attempts'] for r in retry_results)
            avg_attempts = total_attempts / len(retry_results) if retry_results else 0
            st.metric("Avg Attempts", f"{avg_attempts:.2f}")
        
        with col3:
            success_count = sum(1 for r in retry_results if r['retry_result']['success'])
            success_rate = (success_count / len(retry_results) * 100) if retry_results else 0
            st.metric("Retry Success", f"{success_rate:.1f}%", f"{success_count}/{len(retry_results)}")
        
        with col4:
            max_reached = len(retry_results) - success_count
            st.metric("Max Retries", max_reached)
        
        # Retry improvement analysis
        st.markdown("**Retry Effectiveness:**")
        
        improved = 0
        for r in retry_results:
            if r['retry_result']['success']:
                improved += 1
        
        improvement_rate = (improved / len(retry_results) * 100) if retry_results else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.progress(improvement_rate / 100)
        with col2:
            st.caption(f"Improved {improved} out of {len(retry_results)} queries that needed retry")
        
        st.markdown("---")
    
    # =====================================================================
    # ERROR ANALYSIS
    # =====================================================================
    st.markdown("### 🔍 Error Analysis")
    
    # Collect validation errors
    all_errors = []
    for r in results:
        if not r['result'].get('final_is_valid', True):
            errors = r['result'].get('step7', {}).get('errors', [])
            all_errors.extend(errors)
    
    if all_errors:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Error Type Distribution:**")
            error_types = {}
            for error in all_errors:
                error_types[error['type']] = error_types.get(error['type'], 0) + 1
            
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(all_errors) * 100)
                st.write(f"• **{error_type}:** {count} ({pct:.1f}%)")
        
        with col2:
            st.markdown("**Error Severity:**")
            severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for error in all_errors:
                severity_counts[error['severity']] = severity_counts.get(error['severity'], 0) + 1
            
            for severity, count in severity_counts.items():
                if count > 0:
                    pct = (count / len(all_errors) * 100)
                    if severity == 'CRITICAL':
                        st.error(f"🔴 {severity}: {count} ({pct:.1f}%)")
                    elif severity == 'HIGH':
                        st.warning(f"🟠 {severity}: {count} ({pct:.1f}%)")
                    else:
                        st.info(f"🟡 {severity}: {count} ({pct:.1f}%)")
    else:
        st.success("✅ No validation errors found!")
    
    st.markdown("---")
    
    # =====================================================================
    # STRATEGY PERFORMANCE
    # =====================================================================
    st.markdown("### 🎲 Generation Strategy Performance")
    
    strategy_stats = {}
    for result in results:
        strategy = result['result']['step5']['strategy'].value
        
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                'count': 0,
                'valid': 0,
                'ex_success': 0,
                'scores': []
            }
        
        strategy_stats[strategy]['count'] += 1
        
        if result['result'].get('final_is_valid', False):
            strategy_stats[strategy]['valid'] += 1
        
        if result['result'].get('step11'):
            if result['result']['step11']['execution_accuracy']:
                strategy_stats[strategy]['ex_success'] += 1
            strategy_stats[strategy]['scores'].append(result['result']['step11']['evaluation_score'])
    
    col1, col2, col3 = st.columns(3)
    
    cols = [col1, col2, col3]
    for i, (strategy, stats) in enumerate(strategy_stats.items()):
        with cols[i % 3]:
            st.markdown(f"**{strategy.replace('_', ' ').title()}**")
            st.write(f"Queries: {stats['count']}")
            
            valid_rate = (stats['valid'] / stats['count'] * 100) if stats['count'] > 0 else 0
            st.write(f"Valid: {stats['valid']}/{stats['count']} ({valid_rate:.1f}%)")
            
            if stats['scores']:
                ex_rate = (stats['ex_success'] / len(stats['scores']) * 100)
                avg_score = sum(stats['scores']) / len(stats['scores'])
                st.write(f"EX=1.0: {stats['ex_success']}/{len(stats['scores'])} ({ex_rate:.1f}%)")
                st.write(f"Avg Score: {avg_score:.3f}")


# ============================================================================
# CSV GENERATION WITH STATISTICS
# ============================================================================

def generate_comprehensive_csv(results: List[Dict], timestamp: str, output_dir: Path) -> Path:
    """Generate comprehensive CSV with per-query results and summary statistics"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =====================================================================
    # SUMMARY STATISTICS SECTION
    # =====================================================================
    summary_data = []
    
    total = len(results)
    valid_count = sum(1 for r in results if r['result'].get('final_is_valid', False))
    
    summary_data.append({
        'Metric': 'Total Queries',
        'Value': total,
        'Percentage': '100.0%'
    })
    
    summary_data.append({
        'Metric': 'Valid SQL',
        'Value': valid_count,
        'Percentage': f"{(valid_count/total*100):.1f}%" if total > 0 else "0%"
    })
    
    # Execution metrics
    if any(r['result'].get('step10_generated') for r in results):
        exec_success = sum(1 for r in results if r['result'].get('step10_generated', {}).get('success', False))
        summary_data.append({
            'Metric': 'Execution Success',
            'Value': exec_success,
            'Percentage': f"{(exec_success/total*100):.1f}%" if total > 0 else "0%"
        })
    
    # Evaluation metrics
    if any(r['result'].get('step11') for r in results):
        ex_acc = sum(1 for r in results if r['result'].get('step11', {}).get('execution_accuracy', False))
        em_match = sum(1 for r in results if r['result'].get('step11', {}).get('exact_set_match', False))
        eval_results = [r for r in results if r['result'].get('step11')]
        avg_score = sum(r['result']['step11']['evaluation_score'] for r in eval_results) / len(eval_results)
        
        summary_data.append({
            'Metric': 'Execution Accuracy (EX=1.0)',
            'Value': ex_acc,
            'Percentage': f"{(ex_acc/total*100):.1f}%" if total > 0 else "0%"
        })
        
        summary_data.append({
            'Metric': 'Exact-Set-Match (EM=1.0)',
            'Value': em_match,
            'Percentage': f"{(em_match/total*100):.1f}%" if total > 0 else "0%"
        })
        
        summary_data.append({
            'Metric': 'Average Composite Score',
            'Value': f"{avg_score:.4f}",
            'Percentage': f"{(avg_score*100):.1f}%"
        })
    
    # Complexity breakdown
    for complexity in ["EASY", "NON_NESTED_COMPLEX", "NESTED_COMPLEX"]:
        complexity_results = [r for r in results if r['result']['step2']['complexity_class'].value == complexity]
        if complexity_results:
            count = len(complexity_results)
            summary_data.append({
                'Metric': f'{complexity} Queries',
                'Value': count,
                'Percentage': f"{(count/total*100):.1f}%" if total > 0 else "0%"
            })
    
    # Retry metrics
    if any(r.get('retry_result') for r in results):
        retry_results = [r for r in results if r.get('retry_result')]
        success_count = sum(1 for r in retry_results if r['retry_result']['success'])
        
        summary_data.append({
            'Metric': 'Queries Requiring Retry',
            'Value': len(retry_results),
            'Percentage': f"{(len(retry_results)/total*100):.1f}%" if total > 0 else "0%"
        })
        
        summary_data.append({
            'Metric': 'Retry Success Rate',
            'Value': success_count,
            'Percentage': f"{(success_count/len(retry_results)*100):.1f}%" if retry_results else "0%"
        })
    
    # =====================================================================
    # PER-QUERY RESULTS SECTION
    # =====================================================================
    query_data = []
    
    for r in results:
        # Basic info
        row = {
            'Index': r['index'],
            'Database': r['example']['db_id'],
            'Question': r['example']['question'],
            'Complexity': r['result']['step2']['complexity_class'].value,
            'Strategy': r['result']['step5']['strategy'].value,
        }
        
        # Schema info
        row['Num_Tables'] = len(r['result']['step1']['schema_links']['tables'])
        row['Num_Columns'] = sum(len(cols) for cols in r['result']['step1']['schema_links']['columns'].values())
        
        # SQL
        row['Generated_SQL'] = r['result'].get('final_sql', '')
        row['Ground_Truth_SQL'] = r['example'].get('query', '')
        
        # Validation
        row['Is_Valid'] = r['result'].get('final_is_valid', False)
        row['Validation_Score'] = r['result'].get('step7', {}).get('validation_score', 0)
        row['Num_Errors'] = len(r['result'].get('step7', {}).get('errors', []))
        row['Num_Warnings'] = len(r['result'].get('step7', {}).get('warnings', []))
        
        # Execution
        if r['result'].get('step10_generated'):
            row['Execution_Success'] = r['result']['step10_generated']['success']
            row['Execution_Time_Sec'] = r['result']['step10_generated']['execution_time']
            row['Execution_Error'] = r['result']['step10_generated'].get('error_message', '')
        else:
            row['Execution_Success'] = None
            row['Execution_Time_Sec'] = None
            row['Execution_Error'] = ''
        
        # Evaluation
        if r['result'].get('step11'):
            row['Execution_Accuracy_EX'] = r['result']['step11']['execution_accuracy']
            row['Exact_Set_Match_EM'] = r['result']['step11']['exact_set_match']
            row['Evaluation_Score'] = r['result']['step11']['evaluation_score']
            
            # Component match
            component_match = r['result']['step11'].get('component_match', {})
            row['Match_SELECT'] = component_match.get('SELECT', False)
            row['Match_FROM'] = component_match.get('FROM', False)
            row['Match_WHERE'] = component_match.get('WHERE', False)
            row['Match_GROUP_BY'] = component_match.get('GROUP_BY', False)
            row['Match_ORDER_BY'] = component_match.get('ORDER_BY', False)
        else:
            row['Execution_Accuracy_EX'] = None
            row['Exact_Set_Match_EM'] = None
            row['Evaluation_Score'] = None
            row['Match_SELECT'] = None
            row['Match_FROM'] = None
            row['Match_WHERE'] = None
            row['Match_GROUP_BY'] = None
            row['Match_ORDER_BY'] = None
        
        # Retry info
        if r.get('retry_result'):
            row['Retry_Attempts'] = r['retry_result']['total_attempts']
            row['Retry_Success'] = r['retry_result']['success']
        else:
            row['Retry_Attempts'] = None
            row['Retry_Success'] = None
        
        query_data.append(row)
    
    # =====================================================================
    # SAVE TO CSV
    # =====================================================================
    
    # Create filename
    csv_filename = f"adapt_sql_results_{timestamp.replace(' ', '_').replace(':', '-')}.csv"
    csv_path = output_dir / csv_filename
    
    # Create DataFrames
    summary_df = pd.DataFrame(summary_data)
    query_df = pd.DataFrame(query_data)
    
    # Write to CSV with summary section first
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(f"# ADAPT-SQL Batch Processing Results\n")
        f.write(f"# Generated: {timestamp}\n")
        f.write(f"# Total Queries: {total}\n")
        f.write(f"#\n")
        f.write(f"# ========================================\n")
        f.write(f"# SUMMARY STATISTICS\n")
        f.write(f"# ========================================\n")
        f.write(f"#\n")
        
        summary_df.to_csv(f, index=False)
        
        f.write(f"#\n")
        f.write(f"# ========================================\n")
        f.write(f"# PER-QUERY RESULTS\n")
        f.write(f"# ========================================\n")
        f.write(f"#\n")
        
        query_df.to_csv(f, index=False)
    
    return csv_path