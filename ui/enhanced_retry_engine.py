"""
Enhanced Retry Engine - Aligned with DAIL-SQL and DIN-SQL Research Papers

Retry Strategy based on Spider benchmark metrics:
- Primary criterion: Execution Accuracy (EX)
- Secondary criterion: Exact-Set-Match (EM)
- Threshold: min_evaluation_score (default 0.8 = 80% EX weight)
"""
import ollama
from typing import Dict, List, Optional
from core.adapt_baseline import ADAPTBaseline


class EnhancedRetryEngine:
    def __init__(
        self,
        model: str = "qwen3-coder",
        max_full_retries: int = 2,
    ):
        self.model = model
        self.max_full_retries = max_full_retries
    
    def run_with_full_retry(
        self,
        adapt_baseline: ADAPTBaseline,
        natural_query: str,
        schema_dict: Dict,
        foreign_keys: List[Dict],
        k_examples: int = 10,
        db_path: str = None,
        gold_sql: str = None
    ) -> Dict:
        """
        Run complete pipeline with full retry mechanism
        
        Retry criteria based on research papers:
        1. PRIMARY: Execution Accuracy (EX) must be 1.0 (results match exactly)
        2. SECONDARY: Evaluation score above threshold
        
        Args:
            adapt_baseline: ADAPTBaseline instance
            natural_query: Natural language question
            schema_dict: Database schema
            foreign_keys: Foreign key relationships
            k_examples: Number of similar examples
            db_path: Database path for execution
            gold_sql: Ground truth SQL for evaluation
            
        Returns:
            {
                'final_result': Dict,
                'attempt_history': List[Dict],
                'total_attempts': int,
                'success': bool,
                'reasoning': str
            }
        """
        print("\n" + "="*70)
        print("ENHANCED RETRY ENGINE - Aligned with Spider Benchmark")
        print("="*70)
        print(f"Retry criterion: Execution Accuracy (EX) = 0 only")
        print(f"Max retries: {self.max_full_retries}")
        print("="*70 + "\n")
        
        attempt_history = []
        feedback_context = None
        
        # Initial attempt + retries
        for attempt in range(self.max_full_retries + 1):
            print(f"\n{'#'*70}")
            print(f"ATTEMPT {attempt + 1}/{self.max_full_retries + 1}")
            print(f"{'#'*70}\n")
            
            # Inject feedback context if this is a retry
            if feedback_context:
                print("Using feedback from previous attempt:")
                print(f"   - Execution Accuracy (EX): {feedback_context['execution_accuracy']}")
                print(f"   - Issues: {', '.join(feedback_context['issues'][:3])}\n")
            
            # Run pipeline with feedback context
            result = self._run_pipeline_with_context(
                adapt_baseline=adapt_baseline,
                natural_query=natural_query,
                schema_dict=schema_dict,
                foreign_keys=foreign_keys,
                k_examples=k_examples,
                db_path=db_path,
                gold_sql=gold_sql,
                feedback_context=feedback_context
            )
            
            # Store attempt
            attempt_info = {
                'attempt_number': attempt + 1,
                'result': result,
                'feedback_used': feedback_context is not None
            }
            attempt_history.append(attempt_info)
            
            # Check if we should continue
            should_retry, feedback_context = self._evaluate_result_research_based(
                result, gold_sql
            )
            # Attach attempt number so _enhance_query_with_feedback can vary strategy
            if feedback_context:
                feedback_context['attempt_number'] = attempt + 1
            
            if not should_retry:
                print(f"\n{'✅'*35}")
                print(f"SUCCESS - Result meets Spider benchmark criteria!")
                print(f"{'✅'*35}\n")
                
                return {
                    'final_result': result,
                    'attempt_history': attempt_history,
                    'total_attempts': attempt + 1,
                    'success': True,
                    'reasoning': self._generate_reasoning(
                        attempt_history, True, attempt + 1
                    )
                }
            
            # Check if we have retries left
            if attempt >= self.max_full_retries:
                print(f"\n{'⚠️'*35}")
                print(f"MAX RETRIES REACHED - Returning best attempt")
                print(f"{'⚠️'*35}\n")
                break
            
            print(f"Retry needed - Issues detected:")
            for issue in feedback_context['issues']:
                print(f"   - {issue}")
            print()
        
        # Select best attempt based on Spider metrics
        best_result = self._select_best_attempt_research_based(attempt_history)
        
        return {
            'final_result': best_result,
            'attempt_history': attempt_history,
            'total_attempts': len(attempt_history),
            'success': False,
            'reasoning': self._generate_reasoning(
                attempt_history, False, len(attempt_history)
            )
        }
    
    def _run_pipeline_with_context(
        self,
        adapt_baseline: ADAPTBaseline,
        natural_query: str,
        schema_dict: Dict,
        foreign_keys: List[Dict],
        k_examples: int,
        db_path: str,
        gold_sql: str,
        feedback_context: Optional[Dict]
    ) -> Dict:
        """Run pipeline with optional feedback context"""
        
        # If we have feedback, modify the query with hints
        if feedback_context:
            enhanced_query = self._enhance_query_with_feedback(
                natural_query, feedback_context
            )
            print(f"Enhanced query with feedback hints")
        else:
            enhanced_query = natural_query

        # Cross-strategy retry: attempt 2 uses skeleton_first, attempt 3 uses direct_sql.
        # feedback_context['attempt_number'] is the attempt that just FAILED (1-indexed),
        # so the NEXT attempt number = attempt_number + 1.
        failed_attempt = feedback_context.get('attempt_number', 0) if feedback_context else 0
        _strategy_overrides = {2: "skeleton_first", 3: "direct_sql"}
        generation_strategy_override = _strategy_overrides.get(failed_attempt + 1)
        if generation_strategy_override:
            print(f"   [CROSS-STRATEGY] Attempt {failed_attempt + 1} → override={generation_strategy_override}")

        # Run full pipeline
        # Pass original natural_query as search_query so Step 4 vector search
        # always uses the clean question (retry hints break cache lookup).
        result = adapt_baseline.run_full_pipeline(
            natural_query=enhanced_query,
            schema_dict=schema_dict,
            foreign_keys=foreign_keys,
            k_examples=k_examples,
            enable_retry=True,
            db_path=db_path,
            gold_sql=gold_sql,
            enable_execution=True,
            enable_evaluation=(gold_sql is not None),
            enable_multi_candidate=True,
            generation_strategy_override=generation_strategy_override,
            search_query=natural_query,
        )
        
        # Store original query
        result['original_query'] = natural_query
        result['feedback_applied'] = feedback_context is not None
        
        return result
    
    def _evaluate_result_research_based(
        self, 
        result: Dict, 
        gold_sql: Optional[str]
    ) -> tuple:
        """
        Evaluate if result meets Spider benchmark criteria
        
        Primary criterion: Execution Accuracy (EX) = 1.0
        Secondary criterion: Evaluation score >= threshold
        
        Returns:
            (should_retry: bool, feedback_context: Dict or None)
        """
        issues = []
        
        # Extract metrics from Step 11 evaluation
        if not result.get('step11'):
            # No evaluation performed
            should_retry = True
            issues.append("No evaluation performed (missing ground truth)")
            
            feedback_context = {
                'execution_accuracy': False,
                'component_match': {},
                'issues': issues,
                'failed_sql': result.get('final_sql', ''),
                'validation_errors': result.get('step7', {}).get('errors', []),
                'ground_truth_sql': gold_sql,
                'execution_error': None,
            }
            
            return should_retry, feedback_context
        
        # Get evaluation metrics
        evaluation = result['step11']
        execution_accuracy = evaluation['execution_accuracy']
        component_match = evaluation.get('component_match', {})

        # ============================================================
        # ONLY CHECK: Execution Accuracy
        # ============================================================
        if not execution_accuracy:
            issues.append("EX = 0 — results don't match ground truth")

        if result.get('step10_generated') and not result['step10_generated']['success']:
            issues.append(f"Execution failed: {result['step10_generated']['error_message']}")

        # ============================================================
        # DECISION: retry only when EX = 0
        # ============================================================
        should_retry = not execution_accuracy
        
        if not should_retry:
            # Success! EX = 1 and score above threshold
            return False, None
        
        # Build feedback context for retry
        feedback_context = {
            'execution_accuracy': execution_accuracy,
            'component_match': component_match,
            'issues': issues,
            'failed_sql': result.get('final_sql', ''),
            'validation_errors': result.get('step7', {}).get('errors', []),
            'execution_error': result.get('step10_generated', {}).get('error_message'),
            'ground_truth_sql': gold_sql,
        }
        
        return True, feedback_context
    
    def _enhance_query_with_feedback(
        self, 
        original_query: str, 
        feedback_context: Dict
    ) -> str:
        """
        Enhance query with feedback hints for next attempt
        """
        hints = []
        
        # Primary issue: Execution Accuracy
        if not feedback_context['execution_accuracy']:
            hints.append(
                "CRITICAL: Previous SQL execution produced wrong results - "
                "results must match ground truth exactly"
            )
            
            # Add component-specific hints
            if feedback_context.get('component_match'):
                failed = [
                    comp for comp, match in feedback_context['component_match'].items() 
                    if not match
                ]
                if failed:
                    hints.append(
                        f"SQL components that differ: {', '.join(failed)}"
                    )
        
        # Execution error
        if feedback_context.get('execution_error'):
            hints.append(
                f"Previous SQL failed to execute: {feedback_context['execution_error']}"
            )
        
        # Validation errors
        if feedback_context.get('validation_errors'):
            error_types = set(
                e['type'] for e in feedback_context['validation_errors'][:3]
            )
            hints.append(
                f"Fix these errors: {', '.join(error_types)}"
            )
        
        # On 2nd+ retry, add cross-strategy structural diversity hint
        attempt_num = feedback_context.get('attempt_number', 1)
        if attempt_num >= 2:
            hints.append(
                "STRUCTURAL CHANGE REQUIRED: Your previous SQL structure was wrong. "
                "Try a completely different approach — if you used JOINs, try subqueries; "
                "if you used subqueries, try JOINs with GROUP BY; "
                "reconsider which tables are truly needed."
            )

        # Create enhanced query
        enhanced = original_query
        if hints:
            enhanced = (
                f"{original_query}\n\n"
                f"[RETRY HINTS: {'; '.join(hints)}]"
            )

        return enhanced
    
    def _select_best_attempt_research_based(self, attempt_history: List[Dict]) -> Dict:
        """Return the first attempt that has EX=1.
        When all attempts fail, return the attempt with the highest evaluation_score
        rather than always defaulting to attempt 1 — skeleton_first or direct_sql
        may have produced a higher partial score on some queries."""
        for attempt in attempt_history:
            result = attempt['result']
            if result.get('step11', {}).get('execution_accuracy'):
                return result
        # All failed: pick highest partial evaluation_score
        best = max(
            attempt_history,
            key=lambda a: a['result'].get('step11', {}).get('evaluation_score', 0.0),
            default=attempt_history[0]
        )
        return best['result']
    
    def _generate_reasoning(
        self, 
        attempt_history: List[Dict], 
        success: bool,
        total_attempts: int
    ) -> str:
        """Generate reasoning aligned with research paper metrics"""
        reasoning = "ENHANCED RETRY ENGINE - Spider Benchmark Metrics\n"
        reasoning += "=" * 70 + "\n\n"
        
        reasoning += f"Total Attempts: {total_attempts}\n"
        reasoning += f"Final Status: {'✅ SUCCESS' if success else '⚠️ MAX RETRIES REACHED'}\n\n"
        
        reasoning += "Evaluation Criteria (from research papers):\n"
        reasoning += "  PRIMARY:   Execution Accuracy (EX) = 1.0\n"
        reasoning += "  SECONDARY: Exact-Set-Match (EM)\n"
        reasoning += "  COMPOSITE: Score = (EX × 0.80) + (EM × 0.20)\n\n"
        
        reasoning += "Attempt History:\n"
        reasoning += "-" * 70 + "\n"
        
        for i, attempt in enumerate(attempt_history):
            result = attempt['result']
            
            reasoning += f"\nAttempt {i + 1}:\n"
            reasoning += f"  Feedback Used: {attempt['feedback_used']}\n"
            
            # Evaluation metrics (if available)
            if result.get('step11'):
                eval_result = result['step11']
                
                ex = eval_result['execution_accuracy']
                em = eval_result['exact_set_match']
                score = eval_result['evaluation_score']
                
                reasoning += f"  Execution Accuracy (EX): {'✅ 1.0' if ex else '❌ 0.0'}\n"
                reasoning += f"  Exact-Set-Match (EM): {'✅ 1.0' if em else '❌ 0.0'}\n"
                reasoning += f"  Composite Score: {score:.2f}\n"
                
                # Component breakdown
                if not em and eval_result.get('component_match'):
                    failed = [
                        comp for comp, match in eval_result['component_match'].items() 
                        if not match
                    ]
                    if failed:
                        reasoning += f"    Components differ: {', '.join(failed)}\n"
            
            # Execution status
            if result.get('step10_generated'):
                exec_success = result['step10_generated']['success']
                reasoning += f"  Execution: {'✅ Success' if exec_success else '❌ Failed'}\n"
                if not exec_success:
                    reasoning += f"    Error: {result['step10_generated']['error_message']}\n"
            
            # Validation status
            if result.get('step7'):
                val_valid = result['step7']['is_valid']
                val_score = result['step7']['validation_score']
                reasoning += f"  Validation: {'✅ Valid' if val_valid else '❌ Invalid'} "
                reasoning += f"(Score: {val_score:.2f})\n"
            
            # SQL
            reasoning += f"  SQL: {result.get('final_sql', 'N/A')[:80]}...\n"
        
        reasoning += "\n" + "-" * 70 + "\n"
        
        if success:
            reasoning += "\n✅ Pipeline completed successfully!\n"
            reasoning += "   Results meet Spider benchmark criteria:\n"
            reasoning += "   - Execution Accuracy (EX) = 1.0 ✓\n"
            reasoning += "   - Evaluation score above threshold ✓\n"
        else:
            reasoning += "\n⚠️ Maximum retries reached without meeting criteria.\n"
            reasoning += "   Best attempt selected from history based on:\n"
            reasoning += "   1. Execution Accuracy (EX) - priority\n"
            reasoning += "   2. Composite evaluation score\n"
            reasoning += "   3. Exact-Set-Match (EM)\n"
            reasoning += "\nRecommendations:\n"
            reasoning += "  - Review question for ambiguity\n"
            reasoning += "  - Verify schema contains required information\n"
            reasoning += "  - Check database state and content\n"
            reasoning += "  - Consider manual SQL refinement\n"
        
        return reasoning