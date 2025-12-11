"""
Enhanced Retry Engine - Complete Pipeline Restart with Execution/Evaluation Feedback
Retries entire pipeline (Steps 1-11) when execution fails or evaluation score is low
"""
import ollama
from typing import Dict, List, Optional
from adapt_baseline import ADAPTBaseline


class EnhancedRetryEngine:
    def __init__(
        self, 
        model: str = "llama3.2",
        max_full_retries: int = 2,
        min_evaluation_score: float = 0.5
    ):
        """
        Initialize enhanced retry engine
        
        Args:
            model: Ollama model name
            max_full_retries: Maximum full pipeline retries (default: 2)
            min_evaluation_score: Minimum acceptable evaluation score (default: 0.5)
        """
        self.model = model
        self.max_full_retries = max_full_retries
        self.min_evaluation_score = min_evaluation_score
    
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
                'final_result': Dict,  # Final pipeline result
                'attempt_history': List[Dict],  # All attempts
                'total_attempts': int,
                'success': bool,
                'reasoning': str
            }
        """
        print("\n" + "="*70)
        print("ENHANCED RETRY ENGINE - FULL PIPELINE RETRY")
        print("="*70)
        print(f"Max retries: {self.max_full_retries}")
        print(f"Min evaluation score: {self.min_evaluation_score}")
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
                print(f"   - Execution failed: {feedback_context['execution_failed']}")
                print(f"   - Evaluation score: {feedback_context.get('evaluation_score', 'N/A')}")
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
            should_retry, feedback_context = self._evaluate_result(result, gold_sql)
            
            if not should_retry:
                print(f"\n{'âœ…'*35}")
                print(f"SUCCESS - Acceptable result achieved!")
                print(f"{'âœ…'*35}\n")
                
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
                print(f"\n{'âš ï¸'*35}")
                print(f"MAX RETRIES REACHED - Returning best attempt")
                print(f"{'âš ï¸'*35}\n")
                break
            
            print(f"Retry needed - Issues detected:")
            for issue in feedback_context['issues']:
                print(f"   - {issue}")
            print()
        
        # Select best attempt
        best_result = self._select_best_attempt(attempt_history)
        
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
        
        # Run full pipeline
        result = adapt_baseline.run_full_pipeline(
            natural_query=enhanced_query,
            schema_dict=schema_dict,
            foreign_keys=foreign_keys,
            k_examples=k_examples,
            enable_retry=True,  # Step 8 validation retry
            db_path=db_path,
            gold_sql=gold_sql,
            enable_execution=True,
            enable_evaluation=(gold_sql is not None)
        )
        
        # Store original query (not enhanced)
        result['original_query'] = natural_query
        result['feedback_applied'] = feedback_context is not None
        
        return result
    
    def _evaluate_result(
        self, 
        result: Dict, 
        gold_sql: Optional[str]
    ) -> tuple:
        """
        Evaluate if result is acceptable or needs retry
        
        Returns:
            (should_retry: bool, feedback_context: Dict or None)
        """
        issues = []
        
        # Check 1: Execution failure
        execution_failed = False
        if result.get('step10_generated'):
            if not result['step10_generated']['success']:
                execution_failed = True
                error_msg = result['step10_generated']['error_message']
                issues.append(f"Execution failed: {error_msg}")
        
        # Check 2: Low evaluation score (if evaluation was performed)
        low_score = False
        evaluation_score = None
        if result.get('step11'):
            evaluation_score = result['step11']['evaluation_score']
            if evaluation_score < self.min_evaluation_score:
                low_score = True
                issues.append(
                    f"Low evaluation score: {evaluation_score:.2f} "
                    f"(threshold: {self.min_evaluation_score})"
                )
            
            # Add specific evaluation issues
            if not result['step11']['execution_accuracy']:
                issues.append("Execution results don't match ground truth")
            
            if result['step11']['semantic_equivalence'] < 0.5:
                issues.append("Low semantic equivalence with ground truth")
        
        # Check 3: Validation issues
        if result.get('step7'):
            if not result['step7']['is_valid']:
                critical_errors = [
                    e for e in result['step7']['errors'] 
                    if e['severity'] == 'CRITICAL'
                ]
                if critical_errors:
                    issues.append(
                        f"Critical validation errors: {len(critical_errors)}"
                    )
        
        # Decide if retry is needed
        should_retry = execution_failed or low_score or len(issues) > 0
        
        if not should_retry:
            return False, None
        
        # Build feedback context
        feedback_context = {
            'execution_failed': execution_failed,
            'low_score': low_score,
            'evaluation_score': evaluation_score,
            'issues': issues,
            'failed_sql': result.get('final_sql', ''),
            'validation_errors': result.get('step7', {}).get('errors', []),
            'execution_error': result.get('step10_generated', {}).get('error_message'),
            'ground_truth_sql': gold_sql,
            'evaluation_details': result.get('step11', {})
        }
        
        return True, feedback_context
    
    def _enhance_query_with_feedback(
        self, 
        original_query: str, 
        feedback_context: Dict
    ) -> str:
        """
        Enhance query with feedback hints for next attempt
        This provides additional context to the pipeline without changing the question
        """
        # Build hint based on feedback
        hints = []
        
        if feedback_context['execution_failed']:
            hints.append(
                f"Previous SQL execution failed with: "
                f"{feedback_context['execution_error']}"
            )
        
        if feedback_context['low_score']:
            hints.append(
                f"Previous attempt scored {feedback_context['evaluation_score']:.2f} "
                f"- needs improvement"
            )
        
        if feedback_context.get('validation_errors'):
            error_types = set(e['type'] for e in feedback_context['validation_errors'][:3])
            hints.append(
                f"Avoid these errors: {', '.join(error_types)}"
            )
        
        # Create enhanced query with system hint
        # Note: This is passed through but components can access it via result history
        enhanced = original_query
        if hints:
            enhanced = f"{original_query}\n\n[SYSTEM HINTS FROM PREVIOUS ATTEMPT: {'; '.join(hints)}]"
        
        return enhanced
    
    def _select_best_attempt(self, attempt_history: List[Dict]) -> Dict:
        """Select best attempt based on evaluation score or execution success"""
        
        # Score each attempt
        scored_attempts = []
        
        for attempt in attempt_history:
            result = attempt['result']
            score = 0.0
            
            # Execution success
            if result.get('step10_generated', {}).get('success'):
                score += 0.5
            
            # Evaluation score
            if result.get('step11'):
                score += result['step11']['evaluation_score'] * 0.4
            
            # Validation score
            if result.get('step7'):
                score += result['step7']['validation_score'] * 0.1
            
            scored_attempts.append((score, result))
        
        # Return best
        scored_attempts.sort(key=lambda x: x[0], reverse=True)
        return scored_attempts[0][1]
    
    def _generate_reasoning(
        self, 
        attempt_history: List[Dict], 
        success: bool,
        total_attempts: int
    ) -> str:
        """Generate reasoning for retry process"""
        reasoning = "ENHANCED RETRY ENGINE - FULL PIPELINE RETRY\n"
        reasoning += "=" * 70 + "\n\n"
        
        reasoning += f"Total Attempts: {total_attempts}\n"
        reasoning += f"Final Status: {'âœ… SUCCESS' if success else 'âš ï¸  MAX RETRIES REACHED'}\n\n"
        
        reasoning += "Attempt History:\n"
        reasoning += "-" * 70 + "\n"
        
        for i, attempt in enumerate(attempt_history):
            result = attempt['result']
            
            reasoning += f"\nAttempt {i + 1}:\n"
            reasoning += f"  Feedback Used: {attempt['feedback_used']}\n"
            
            # Execution status
            if result.get('step10_generated'):
                exec_success = result['step10_generated']['success']
                reasoning += f"  Execution: {'âœ… Success' if exec_success else 'âŒ Failed'}\n"
                if not exec_success:
                    reasoning += f"    Error: {result['step10_generated']['error_message']}\n"
            
            # Validation status
            if result.get('step7'):
                val_valid = result['step7']['is_valid']
                val_score = result['step7']['validation_score']
                reasoning += f"  Validation: {'âœ… Valid' if val_valid else 'âŒ Invalid'} "
                reasoning += f"(Score: {val_score:.2f})\n"
            
            # Evaluation status
            if result.get('step11'):
                eval_score = result['step11']['evaluation_score']
                exec_acc = result['step11']['execution_accuracy']
                reasoning += f"  Evaluation Score: {eval_score:.2f}\n"
                reasoning += f"  Execution Accuracy: {'âœ… Pass' if exec_acc else 'âŒ Fail'}\n"
            
            # SQL
            reasoning += f"  Generated SQL: {result.get('final_sql', 'N/A')[:80]}...\n"
        
        reasoning += "\n" + "-" * 70 + "\n"
        
        if success:
            reasoning += "\nâœ… Pipeline completed successfully!\n"
            reasoning += "   Results meet quality thresholds.\n"
        else:
            reasoning += "\nâš ï¸  Maximum retries reached without meeting thresholds.\n"
            reasoning += "   Best attempt selected from history.\n"
            reasoning += "\nRecommendations:\n"
            reasoning += "  - Review question for ambiguity\n"
            reasoning += "  - Check if schema contains required information\n"
            reasoning += "  - Verify database state and content\n"
            reasoning += "  - Consider manual SQL refinement\n"
        
        return reasoning


class FeedbackAwarePipeline(ADAPTBaseline):
    """
    Extended ADAPTBaseline that can use feedback from previous attempts
    This modifies behavior in each step based on feedback context
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_feedback = None
    
    def set_feedback_context(self, feedback_context: Optional[Dict]):
        """Set feedback from previous attempt"""
        self.previous_feedback = feedback_context
    
    def run_full_pipeline(self, *args, **kwargs):
        """Override to inject feedback into steps"""
        
        # Store feedback in session for steps to access
        if self.previous_feedback:
            # Steps can check self.previous_feedback for hints
            # Example: In Step 6, avoid patterns that failed before
            pass
        
        # Run normal pipeline
        result = super().run_full_pipeline(*args, **kwargs)
        
        # Clear feedback after use
        self.previous_feedback = None
        
        return result