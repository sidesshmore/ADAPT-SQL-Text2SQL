"""
STEP 10: Execute and Compare
Executes generated SQL on the database and handles errors gracefully
"""
import sqlite3
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path


class DatabaseManager:
    def __init__(self, timeout: int = 30):
        """
        Initialize database manager
        
        Args:
            timeout: Query execution timeout in seconds
        """
        self.timeout = timeout
    
    def execute_query(
        self,
        sql: str,
        db_path: str
    ) -> Dict:
        """
        STEP 10: Execute and Compare
        
        Args:
            sql: SQL query to execute
            db_path: Path to SQLite database
            
        Returns:
            {
                'success': bool,
                'result_df': pd.DataFrame or None,
                'result_rows': List[tuple] or None,
                'column_names': List[str] or None,
                'error_message': str or None,
                'execution_time': float,
                'reasoning': str
            }
        """
        print(f"\n{'='*60}")
        print("STEP 10: EXECUTE AND COMPARE")
        print(f"{'='*60}\n")
        
        print(f"Database: {db_path}")
        print(f"SQL Query: {sql[:100]}{'...' if len(sql) > 100 else ''}")
        
        # Check if database exists
        if not Path(db_path).exists():
            error_msg = f"Database file not found: {db_path}"
            print(f"❌ {error_msg}")
            
            return {
                'success': False,
                'result_df': None,
                'result_rows': None,
                'column_names': None,
                'error_message': error_msg,
                'execution_time': 0.0,
                'reasoning': self._generate_reasoning(
                    sql, db_path, False, error_message=error_msg
                )
            }
        
        # Execute query
        print("10.1: Executing SQL query...")
        
        import time
        start_time = time.time()
        
        try:
            # Connect to database
            conn = sqlite3.connect(
                db_path,
                timeout=self.timeout,
                check_same_thread=False
            )
            
            # Set timeout
            conn.execute(f"PRAGMA busy_timeout = {self.timeout * 1000}")
            
            # Execute query
            cursor = conn.cursor()
            cursor.execute(sql)
            
            # Fetch results
            result_rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            execution_time = time.time() - start_time
            
            # Create DataFrame
            if result_rows and column_names:
                result_df = pd.DataFrame(result_rows, columns=column_names)
            else:
                result_df = pd.DataFrame()
            
            # Close connection
            conn.close()
            
            print(f"   ✅ Query executed successfully")
            print(f"   Rows returned: {len(result_rows)}")
            print(f"   Columns: {len(column_names)}")
            print(f"   Execution time: {execution_time:.3f}s")
            
            reasoning = self._generate_reasoning(
                sql, db_path, True, result_df=result_df, 
                execution_time=execution_time
            )
            
            print(f"\n{'='*60}")
            print("STEP 10 COMPLETED ✓")
            print(f"{'='*60}\n")
            
            return {
                'success': True,
                'result_df': result_df,
                'result_rows': result_rows,
                'column_names': column_names,
                'error_message': None,
                'execution_time': execution_time,
                'reasoning': reasoning
            }
            
        except sqlite3.OperationalError as e:
            execution_time = time.time() - start_time
            error_msg = f"Operational error: {str(e)}"
            
            print(f"   ❌ {error_msg}")
            print(f"   Execution time: {execution_time:.3f}s")
            
            reasoning = self._generate_reasoning(
                sql, db_path, False, error_message=error_msg,
                execution_time=execution_time
            )
            
            print(f"\n{'='*60}")
            print("STEP 10 COMPLETED - EXECUTION FAILED")
            print(f"{'='*60}\n")
            
            return {
                'success': False,
                'result_df': None,
                'result_rows': None,
                'column_names': None,
                'error_message': error_msg,
                'execution_time': execution_time,
                'reasoning': reasoning
            }
            
        except sqlite3.DatabaseError as e:
            execution_time = time.time() - start_time
            error_msg = f"Database error: {str(e)}"
            
            print(f"   ❌ {error_msg}")
            print(f"   Execution time: {execution_time:.3f}s")
            
            reasoning = self._generate_reasoning(
                sql, db_path, False, error_message=error_msg,
                execution_time=execution_time
            )
            
            print(f"\n{'='*60}")
            print("STEP 10 COMPLETED - EXECUTION FAILED")
            print(f"{'='*60}\n")
            
            return {
                'success': False,
                'result_df': None,
                'result_rows': None,
                'column_names': None,
                'error_message': error_msg,
                'execution_time': execution_time,
                'reasoning': reasoning
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            
            print(f"   ❌ {error_msg}")
            print(f"   Execution time: {execution_time:.3f}s")
            
            reasoning = self._generate_reasoning(
                sql, db_path, False, error_message=error_msg,
                execution_time=execution_time
            )
            
            print(f"\n{'='*60}")
            print("STEP 10 COMPLETED - EXECUTION FAILED")
            print(f"{'='*60}\n")
            
            return {
                'success': False,
                'result_df': None,
                'result_rows': None,
                'column_names': None,
                'error_message': error_msg,
                'execution_time': execution_time,
                'reasoning': reasoning
            }
    
    def execute_both_queries(
        self,
        generated_sql: str,
        gold_sql: str,
        db_path: str
    ) -> Dict:
        """
        Execute both generated and gold SQL queries for comparison
        
        Args:
            generated_sql: Generated SQL query
            gold_sql: Ground truth SQL query
            db_path: Path to SQLite database
            
        Returns:
            {
                'generated_result': Dict (from execute_query),
                'gold_result': Dict (from execute_query),
                'comparison': str
            }
        """
        print(f"\n{'='*60}")
        print("EXECUTING BOTH QUERIES FOR COMPARISON")
        print(f"{'='*60}\n")
        
        # Execute generated SQL
        print("🔹 Executing generated SQL...")
        generated_result = self.execute_query(generated_sql, db_path)
        
        print("\n" + "-"*60 + "\n")
        
        # Execute gold SQL
        print("🔹 Executing ground truth SQL...")
        gold_result = self.execute_query(gold_sql, db_path)
        
        # Compare results
        comparison = self._compare_results(generated_result, gold_result)
        
        print(f"\n{'='*60}")
        print("COMPARISON COMPLETE")
        print(f"{'='*60}\n")
        
        return {
            'generated_result': generated_result,
            'gold_result': gold_result,
            'comparison': comparison
        }
    
    def _compare_results(
        self,
        generated_result: Dict,
        gold_result: Dict
    ) -> str:
        """Compare execution results"""
        comparison = "EXECUTION COMPARISON\n"
        comparison += "=" * 50 + "\n\n"
        
        # Check if both executed successfully
        if not generated_result['success']:
            comparison += "❌ Generated SQL failed to execute\n"
            comparison += f"   Error: {generated_result['error_message']}\n"
        else:
            comparison += "✅ Generated SQL executed successfully\n"
            comparison += f"   Rows: {len(generated_result['result_df'])}\n"
            comparison += f"   Columns: {len(generated_result['column_names'])}\n"
        
        comparison += "\n"
        
        if not gold_result['success']:
            comparison += "❌ Ground truth SQL failed to execute\n"
            comparison += f"   Error: {gold_result['error_message']}\n"
        else:
            comparison += "✅ Ground truth SQL executed successfully\n"
            comparison += f"   Rows: {len(gold_result['result_df'])}\n"
            comparison += f"   Columns: {len(gold_result['column_names'])}\n"
        
        comparison += "\n"
        
        # If both succeeded, compare results
        if generated_result['success'] and gold_result['success']:
            gen_df = generated_result['result_df']
            gold_df = gold_result['result_df']
            
            # Compare shapes
            if gen_df.shape == gold_df.shape:
                comparison += "✅ Same result shape\n"
            else:
                comparison += "❌ Different result shapes\n"
                comparison += f"   Generated: {gen_df.shape}\n"
                comparison += f"   Ground truth: {gold_df.shape}\n"
            
            comparison += "\n"
            
            # Compare column names
            gen_cols = set(gen_df.columns)
            gold_cols = set(gold_df.columns)
            
            if gen_cols == gold_cols:
                comparison += "✅ Same column names\n"
            else:
                comparison += "❌ Different column names\n"
                if gen_cols - gold_cols:
                    comparison += f"   Extra in generated: {gen_cols - gold_cols}\n"
                if gold_cols - gen_cols:
                    comparison += f"   Missing in generated: {gold_cols - gen_cols}\n"
        
        return comparison
    
    def _generate_reasoning(
        self,
        sql: str,
        db_path: str,
        success: bool,
        result_df: Optional[pd.DataFrame] = None,
        error_message: Optional[str] = None,
        execution_time: float = 0.0
    ) -> str:
        """Generate reasoning for execution result"""
        reasoning = "STEP 10: EXECUTE AND COMPARE\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Database: {db_path}\n\n"
        
        reasoning += "SQL Query:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Execution Time: {execution_time:.3f}s\n"
        reasoning += f"Success: {'✅ Yes' if success else '❌ No'}\n\n"
        
        if success and result_df is not None:
            reasoning += "Execution Results:\n"
            reasoning += f"  • Rows returned: {len(result_df)}\n"
            reasoning += f"  • Columns: {len(result_df.columns)}\n"
            reasoning += f"  • Column names: {', '.join(result_df.columns)}\n\n"
            
            if len(result_df) > 0:
                reasoning += "Sample Results (first 5 rows):\n"
                reasoning += "-" * 50 + "\n"
                reasoning += result_df.head(5).to_string(index=False) + "\n"
                reasoning += "-" * 50 + "\n"
        
        elif not success and error_message:
            reasoning += "Execution Failed:\n"
            reasoning += f"  Error: {error_message}\n\n"
            
            reasoning += "Possible Causes:\n"
            if "no such table" in error_message.lower():
                reasoning += "  • Table name is incorrect or doesn't exist\n"
            elif "no such column" in error_message.lower():
                reasoning += "  • Column name is incorrect or doesn't exist\n"
            elif "syntax error" in error_message.lower():
                reasoning += "  • SQL syntax is invalid\n"
            elif "ambiguous" in error_message.lower():
                reasoning += "  • Column reference is ambiguous (needs table prefix)\n"
            else:
                reasoning += "  • Check SQL syntax and schema correctness\n"
        
        return reasoning