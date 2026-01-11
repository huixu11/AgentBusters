"""
BizFinBench.v2 dataset evaluator.

Evaluates predictions against BizFinBench.v2 ground truth using task-specific
evaluation strategies:
- Numerical tasks: Match with tolerance
- Ordering tasks: Exact sequence match
- Classification tasks: Label match
- Open-ended tasks: LLM-based judgment
"""

import logging
import re
from typing import Optional

from evaluators.base import BaseDatasetEvaluator, EvalResult

logger = logging.getLogger(__name__)


class BizFinBenchEvaluator(BaseDatasetEvaluator):
    """
    Evaluator for HiThink BizFinBench.v2 dataset.
    
    Supports different evaluation strategies per task type:
        - financial_quantitative_computation: Numerical match ±tolerance
        - event_logic_reasoning: Exact sequence match
        - user_sentiment_analysis: Classification match
        - stock_price_predict: Numerical match
        - Others: Normalized string match
    """
    
    name = "bizfinbench"
    
    # Default numerical tolerance (1%)
    DEFAULT_TOLERANCE = 0.01
    
    def __init__(self, tolerance: float = None):
        """
        Initialize BizFinBench evaluator.
        
        Args:
            tolerance: Numerical tolerance for quantitative tasks.
                       If None, uses DEFAULT_TOLERANCE (currently 0.01 = 1%).
        """
        self.tolerance = tolerance or self.DEFAULT_TOLERANCE
    
    def evaluate(
        self,
        predicted: str,
        expected: str,
        task_type: str = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate predicted answer against expected.
        
        Args:
            predicted: Model's predicted answer
            expected: Ground truth answer
            task_type: BizFinBench task type (determines evaluation strategy)
            **kwargs: Additional parameters
            
        Returns:
            EvalResult with score and details
        """
        if not predicted or not expected:
            return EvalResult(
                score=0.0,
                feedback="Empty prediction or expected answer",
                details={"predicted": predicted, "expected": expected}
            )
        
        # Normalize inputs
        predicted = predicted.strip()
        expected = expected.strip()
        
        # Route to task-specific evaluator
        if task_type == "financial_quantitative_computation":
            return self._eval_numerical(predicted, expected)
        elif task_type == "event_logic_reasoning":
            return self._eval_exact_sequence(predicted, expected)
        elif task_type == "user_sentiment_analysis":
            return self._eval_classification(predicted, expected)
        elif task_type == "stock_price_predict":
            return self._eval_numerical(predicted, expected)
        elif task_type == "conterfactual":
            return self._eval_normalized_match(predicted, expected)
        else:
            # Default: normalized string match
            return self._eval_normalized_match(predicted, expected)
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text."""
        # Try to parse as float directly
        try:
            return float(text)
        except ValueError:
            pass
        
        # Look for numbers in text (handle percentages, commas, etc.)
        patterns = [
            r'(-?\d+\.?\d*)\s*%',  # Percentage
            r'(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # Number with commas
            r'(-?\d+\.?\d*)',  # Simple number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                num_str = match.group(1).replace(',', '')
                try:
                    return float(num_str)
                except ValueError:
                    continue
        
        return None
    
    def _eval_numerical(self, predicted: str, expected: str) -> EvalResult:
        """
        Evaluate numerical answer with tolerance.
        
        Args:
            predicted: Predicted numerical answer
            expected: Expected numerical answer
            
        Returns:
            EvalResult (1.0 if within tolerance, 0.0 otherwise)
        """
        pred_num = self._extract_number(predicted)
        exp_num = self._extract_number(expected)
        
        if pred_num is None:
            return EvalResult(
                score=0.0,
                feedback=f"Could not extract number from prediction: '{predicted}'",
                details={"predicted": predicted, "expected": expected}
            )
        
        if exp_num is None:
            return EvalResult(
                score=0.0,
                feedback=f"Could not extract number from expected: '{expected}'",
                details={"predicted": predicted, "expected": expected}
            )
        
        # Calculate relative error
        if exp_num == 0:
            is_correct = abs(pred_num) < self.tolerance
        else:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
            is_correct = relative_error <= self.tolerance
        
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            correct_count=1 if is_correct else 0,
            total_count=1,
            feedback=f"Predicted: {pred_num}, Expected: {exp_num}, Tolerance: {self.tolerance}",
            details={
                "predicted_num": pred_num,
                "expected_num": exp_num,
                "is_correct": is_correct,
                "tolerance": self.tolerance,
            }
        )
    
    def _eval_exact_sequence(self, predicted: str, expected: str) -> EvalResult:
        """
        Evaluate sequence/ordering answer (exact match).
        
        Expected format: "2,1,4,3" or similar
        
        Args:
            predicted: Predicted sequence
            expected: Expected sequence
            
        Returns:
            EvalResult (1.0 if exact match, 0.0 otherwise)
        """
        # Normalize: remove spaces, extract comma-separated values
        pred_clean = re.sub(r'\s+', '', predicted)
        exp_clean = re.sub(r'\s+', '', expected)
        
        # Extract sequence (handle JSON format if present)
        pred_match = re.search(r'[\d,]+', pred_clean)
        exp_match = re.search(r'[\d,]+', exp_clean)
        
        if pred_match:
            pred_seq = pred_match.group()
        else:
            pred_seq = pred_clean
        
        if exp_match:
            exp_seq = exp_match.group()
        else:
            exp_seq = exp_clean
        
        is_correct = pred_seq == exp_seq
        
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            correct_count=1 if is_correct else 0,
            total_count=1,
            feedback=f"Predicted: '{pred_seq}', Expected: '{exp_seq}'",
            details={
                "predicted_seq": pred_seq,
                "expected_seq": exp_seq,
                "is_correct": is_correct,
            }
        )
    
    def _eval_classification(self, predicted: str, expected: str) -> EvalResult:
        """
        Evaluate classification answer.
        
        Args:
            predicted: Predicted label
            expected: Expected label
            
        Returns:
            EvalResult (1.0 if match, 0.0 otherwise)
        """
        # Normalize: lowercase, strip
        pred_norm = predicted.lower().strip()
        exp_norm = expected.lower().strip()
        
        is_correct = pred_norm == exp_norm
        
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            correct_count=1 if is_correct else 0,
            total_count=1,
            feedback=f"Predicted: '{pred_norm}', Expected: '{exp_norm}'",
            details={
                "predicted": pred_norm,
                "expected": exp_norm,
                "is_correct": is_correct,
            }
        )
    
    def _eval_normalized_match(self, predicted: str, expected: str) -> EvalResult:
        """
        Evaluate with normalized string matching.
        
        Handles minor formatting differences.
        
        Args:
            predicted: Predicted answer
            expected: Expected answer
            
        Returns:
            EvalResult with similarity score
        """
        # Normalize both strings
        pred_norm = self._normalize_text(predicted)
        exp_norm = self._normalize_text(expected)
        
        is_correct = pred_norm == exp_norm
        
        # Calculate partial match score using simple containment
        if is_correct:
            score = 1.0
        elif exp_norm in pred_norm or pred_norm in exp_norm:
            score = 0.5
        else:
            score = 0.0
        
        return EvalResult(
            score=score,
            correct_count=1 if is_correct else 0,
            total_count=1,
            feedback=f"Exact match: {is_correct}",
            details={
                "predicted_normalized": pred_norm[:100],
                "expected_normalized": exp_norm[:100],
                "is_exact_match": is_correct,
            }
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common punctuation
        text = re.sub(r'[.,;:!?\'"()\[\]{}]', '', text)
        return text
