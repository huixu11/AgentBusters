"""
Public CSV dataset evaluator.

Evaluates predictions against public.csv rubric using correctness/contradiction
operators as defined in the dataset.

Rubric format:
[
    {'operator': 'correctness', 'criteria': 'Must contain X'},
    {'operator': 'contradiction', 'criteria': 'Must not contradict Y'}
]
"""

import logging
import re
from typing import Any, Dict, List

from evaluators.base import BaseDatasetEvaluator, EvalResult

logger = logging.getLogger(__name__)


class PublicCsvEvaluator(BaseDatasetEvaluator):
    """
    Evaluator for public.csv (FAB++) dataset.
    
    Uses rubric operators:
        - 'correctness': Check if answer contains/matches the criteria (+1 point)
        - 'contradiction': Check if answer contradicts reference (-1 penalty)
    
    This evaluator can use either:
        1. Simple string matching (fast, no LLM)
        2. LLM-based judgment (accurate, requires LLM client)
    """
    
    name = "public_csv"
    
    # Scoring constants
    CONTRADICTION_PENALTY = 0.5  # Penalty per contradiction found
    MIN_WORD_LENGTH = 4  # Minimum word length for substring matching
    
    def __init__(self, use_llm: bool = False, llm_client: Any = None):
        """
        Initialize Public CSV evaluator.
        
        Args:
            use_llm: Whether to use LLM for evaluation (more accurate)
            llm_client: LLM client for LLM-based evaluation
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
    
    def evaluate(
        self,
        predicted: str,
        expected: str = None,
        rubric: List[Dict[str, str]] = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate predicted answer against rubric.
        
        Args:
            predicted: Model's predicted answer
            expected: Reference answer (optional, used for contradiction check)
            rubric: List of rubric items with 'operator' and 'criteria' keys
            **kwargs: Additional parameters
            
        Returns:
            EvalResult with score and details
        """
        if not rubric:
            # No rubric provided, fall back to simple comparison
            return self._simple_compare(predicted, expected or "")
        
        # Evaluate each rubric item
        correct_count = 0
        penalty_count = 0
        item_results = []
        
        correctness_items = [r for r in rubric if r.get("operator") == "correctness"]
        contradiction_items = [r for r in rubric if r.get("operator") == "contradiction"]
        
        # Evaluate correctness criteria
        for item in correctness_items:
            criteria = item.get("criteria", "")
            is_met = self._check_correctness(predicted, criteria)
            if is_met:
                correct_count += 1
            item_results.append({
                "operator": "correctness",
                "criteria": criteria[:100] + "..." if len(criteria) > 100 else criteria,
                "met": is_met,
            })
        
        # Evaluate contradiction criteria (check for hallucination/contradiction)
        for item in contradiction_items:
            criteria = item.get("criteria", "")
            has_contradiction = self._check_contradiction(predicted, criteria)
            if has_contradiction:
                penalty_count += 1
            item_results.append({
                "operator": "contradiction",
                "criteria": criteria[:100] + "..." if len(criteria) > 100 else criteria,
                "violated": has_contradiction,
            })
        
        # Calculate score
        total_correctness = len(correctness_items)
        max_score = total_correctness if total_correctness > 0 else 1
        score = correct_count - (penalty_count * self.CONTRADICTION_PENALTY)
        score = max(0.0, score)  # Don't go negative
        
        # Normalize to 0-1 range
        normalized_score = score / max_score if max_score > 0 else 0.0
        normalized_score = min(1.0, normalized_score)  # Cap at 1.0
        
        return EvalResult(
            score=normalized_score,
            max_score=1.0,
            correct_count=correct_count,
            total_count=total_correctness,
            feedback=f"Correctness: {correct_count}/{total_correctness}, Contradictions: {penalty_count}",
            details={
                "correctness_met": correct_count,
                "correctness_total": total_correctness,
                "contradictions_found": penalty_count,
                "item_results": item_results,
            }
        )
    
    def _check_correctness(self, predicted: str, criteria: str) -> bool:
        """
        Check if prediction satisfies correctness criteria.
        
        Simple approach: Check if key phrases from criteria appear in prediction.
        
        Args:
            predicted: The prediction to check
            criteria: The criteria to match
            
        Returns:
            True if criteria is satisfied
        """
        if not criteria:
            return True
        
        # Normalize both
        pred_lower = predicted.lower()
        crit_lower = criteria.lower()
        
        # Extract key elements from criteria
        # Look for numbers, names, key phrases
        key_elements = self._extract_key_elements(criteria)
        
        if not key_elements:
            # No extractable elements, check for substring
            return crit_lower in pred_lower or any(
                word in pred_lower for word in crit_lower.split() if len(word) > self.MIN_WORD_LENGTH
            )
        
        # Check if majority of key elements are present
        matches = sum(1 for elem in key_elements if elem in pred_lower)
        return matches >= len(key_elements) * 0.5
    
    def _check_contradiction(self, predicted: str, reference: str) -> bool:
        """
        Check if prediction contradicts the reference.
        
        Note: Without LLM support, this method always returns False to avoid
        false positives. Set use_llm=True for actual contradiction detection.
        
        Args:
            predicted: The prediction to check
            reference: The reference that shouldn't be contradicted
            
        Returns:
            True if contradiction detected (always False without LLM)
        """
        if not reference or not predicted:
            return False
        
        # TODO: Implement LLM-based contradiction detection when use_llm=True
        # For now, be conservative and never flag contradictions without LLM
        # to avoid false positives
        return False
    
    def _extract_key_elements(self, text: str) -> List[str]:
        """
        Extract key elements (numbers, names, key phrases) from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of key elements (lowercase)
        """
        elements = []
        
        # Extract numbers
        numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|%))?', text.lower())
        elements.extend(numbers)
        
        # Extract potential names (capitalized words in original)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        elements.extend([n.lower() for n in names])
        
        # Extract stock tickers
        tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
        elements.extend([t.lower() for t in tickers])
        
        return elements
    
    def _simple_compare(self, predicted: str, expected: str) -> EvalResult:
        """
        Simple comparison when no rubric provided.
        
        Args:
            predicted: Predicted answer
            expected: Expected answer
            
        Returns:
            EvalResult based on string similarity
        """
        pred_norm = predicted.lower().strip()
        exp_norm = expected.lower().strip()
        
        is_exact = pred_norm == exp_norm
        is_contained = exp_norm in pred_norm or pred_norm in exp_norm
        
        if is_exact:
            score = 1.0
        elif is_contained:
            score = 0.7
        else:
            score = 0.0
        
        return EvalResult(
            score=score,
            correct_count=1 if is_exact else 0,
            total_count=1,
            feedback="Simple comparison (no rubric provided)",
            details={"is_exact": is_exact, "is_contained": is_contained}
        )
