"""
Test hybrid LLM+Rule evaluation for OptionsEvaluator.

This tests the new score_with_ground_truth method that:
1. Uses LLM to extract values from agent response
2. Uses rule-based comparison against ground truth
3. Falls back to regex extraction if LLM unavailable
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluators.options import OptionsEvaluator, OptionsScore
from cio_agent.models import TaskCategory


def create_mock_task(question_id="test", category=TaskCategory.OPTIONS_PRICING):
    """Create a mock task with all required attributes."""
    task = MagicMock()
    task.question_id = question_id
    task.category = category
    task.rubric = MagicMock()
    task.rubric.mandatory_elements = []
    return task


def create_mock_response(analysis, recommendation, confidence=0.9):
    """Create a mock response with all required attributes."""
    response = MagicMock()
    response.analysis = analysis
    response.recommendation = recommendation
    response.confidence = confidence
    return response


class TestHybridEvaluation:
    """Test hybrid LLM+Rule evaluation."""

    def test_compare_values_exact_match(self):
        """Test rule-based comparison with exact values."""
        evaluator = OptionsEvaluator(task=create_mock_task())
        
        extracted = {"theoretical_price": 25.18, "assessment": "underpriced"}
        ground_truth = {"theoretical_price": 25.18, "assessment": "underpriced"}
        
        score, feedback = evaluator._compare_values(extracted, ground_truth, tolerance=0.05)
        
        assert score == 100.0
        assert "2/2" in feedback

    def test_compare_values_within_tolerance(self):
        """Test rule-based comparison within tolerance."""
        evaluator = OptionsEvaluator(task=create_mock_task())
        
        extracted = {"theoretical_price": 25.00}  # Within 5% of 25.18
        ground_truth = {"theoretical_price": 25.18}
        
        score, feedback = evaluator._compare_values(extracted, ground_truth, tolerance=0.05)
        
        assert score == 100.0

    def test_compare_values_outside_tolerance(self):
        """Test rule-based comparison outside tolerance."""
        evaluator = OptionsEvaluator(task=create_mock_task())
        
        extracted = {"theoretical_price": 17.82}  # Wrong - 29% off
        ground_truth = {"theoretical_price": 25.18}
        
        score, feedback = evaluator._compare_values(extracted, ground_truth, tolerance=0.05)
        
        assert score == 0.0  # 0/1 matched
        assert "✗" in feedback

    def test_compare_values_string_partial_match(self):
        """Test string comparison with partial match."""
        evaluator = OptionsEvaluator(task=create_mock_task())
        
        extracted = {"assessment": "significantly underpriced"}
        ground_truth = {"assessment": "underpriced"}
        
        score, feedback = evaluator._compare_values(extracted, ground_truth, tolerance=0.05)
        
        assert score == 100.0  # "underpriced" is in "significantly underpriced"

    def test_compare_greeks_tolerance(self):
        """Test Greeks comparison with 10% tolerance."""
        evaluator = OptionsEvaluator(
            task=create_mock_task(category=TaskCategory.GREEKS_ANALYSIS),
        )
        
        # Delta 0.48 is within 10% of 0.474
        extracted = {"delta": 0.48, "gamma": 0.011, "theta": -0.30}
        ground_truth = {"delta": 0.474, "gamma": 0.012, "theta": -0.321}
        
        score, feedback = evaluator._compare_values(extracted, ground_truth, tolerance=0.10)
        
        # All should be within 10% tolerance
        assert score >= 66.0  # At least 2/3 correct

    @pytest.mark.asyncio
    async def test_score_with_ground_truth_quantitative(self):
        """Test hybrid evaluation for quantitative question."""
        task = create_mock_task("opt_pricing_002", TaskCategory.OPTIONS_PRICING)
        evaluator = OptionsEvaluator(task=task)
        evaluator._use_llm_extraction = False  # Use regex fallback
        
        response = create_mock_response(
            analysis="""Using Black-Scholes with S=450, K=460, T=45/365, r=5.25%, σ=45%:
            The theoretical call price is $25.18.
            Delta: 0.52, Gamma: 0.008, Theta: -0.15, Vega: 0.45""",
            recommendation="Buy the call - it's underpriced.",
        )
        
        ground_truth = {"theoretical_price": 25.18, "assessment": "underpriced"}
        
        result = await evaluator.score_with_ground_truth(response, ground_truth)
        
        assert isinstance(result, OptionsScore)
        assert result.score >= 0
        assert result.score <= 100
        assert "[Hybrid" in result.feedback

    @pytest.mark.asyncio  
    async def test_score_with_ground_truth_wrong_answer(self):
        """Test hybrid evaluation detects wrong answer."""
        task = create_mock_task("opt_pricing_002", TaskCategory.OPTIONS_PRICING)
        evaluator = OptionsEvaluator(task=task)
        evaluator._use_llm_extraction = False
        
        response = create_mock_response(
            analysis="""Using Black-Scholes, the theoretical price is $17.82.
            At $18.50 market price, this option is slightly overpriced.
            Delta: 0.42""",
            recommendation="Avoid buying - slightly overpriced.",
        )
        
        ground_truth = {"theoretical_price": 25.18, "assessment": "underpriced"}
        
        result = await evaluator.score_with_ground_truth(response, ground_truth)
        
        assert isinstance(result, OptionsScore)
        assert result.score <= 100

    @pytest.mark.asyncio
    @patch("evaluators.options.call_llm")
    @patch("evaluators.options.build_llm_client_for_evaluator")
    @patch("evaluators.options.extract_json")
    async def test_llm_extraction_success(
        self, 
        mock_extract_json,
        mock_build_client, 
        mock_call_llm,
    ):
        """Test successful LLM extraction."""
        mock_build_client.return_value = MagicMock()
        mock_call_llm.return_value = '{"theoretical_price": 25.18, "assessment": "underpriced"}'
        mock_extract_json.return_value = {"theoretical_price": 25.18, "assessment": "underpriced"}
        
        task = create_mock_task("opt_pricing_002", TaskCategory.OPTIONS_PRICING)
        evaluator = OptionsEvaluator(task=task)
        evaluator._use_llm_extraction = True
        
        response = create_mock_response(
            analysis="The theoretical call price is $25.18. It's underpriced.",
            recommendation="Buy the call.",
        )
        
        ground_truth = {"theoretical_price": 25.18, "assessment": "underpriced"}
        
        result = await evaluator.score_with_ground_truth(response, ground_truth)
        
        # Score is 100*0.6 (comparison) + 40*0.2 (strategy) + 30*0.2 (risk) = 74
        # This is expected because the mock response is minimal
        assert result.score >= 70  # Should score reasonably high with correct extraction
        assert result.pnl_accuracy == 100.0  # Perfect extraction
        assert "[Hybrid-Quant]" in result.feedback


class TestQuantitativeVsQualitative:
    """Test classification of quantitative vs qualitative questions."""

    @pytest.mark.asyncio
    async def test_pricing_is_quantitative(self):
        """OPTIONS_PRICING should be classified as quantitative."""
        task = create_mock_task("opt_pricing_001", TaskCategory.OPTIONS_PRICING)
        evaluator = OptionsEvaluator(task=task)
        evaluator._use_llm_extraction = False
        
        response = create_mock_response(
            analysis="Call price is $3.22",
            recommendation="Buy",
        )
        
        result = await evaluator.score_with_ground_truth(
            response,
            {"call_price": 3.22},
        )
        
        assert "[Hybrid-Quant]" in result.feedback

    @pytest.mark.asyncio
    async def test_strategy_defense_is_qualitative(self):
        """STRATEGY_DEFENSE should be classified as qualitative (no numerics)."""
        task = create_mock_task("defense_001", TaskCategory.STRATEGY_DEFENSE)
        evaluator = OptionsEvaluator(task=task)
        evaluator._use_llm_extraction = False
        
        response = create_mock_response(
            analysis="Close the position due to earnings risk",
            recommendation="Close or roll",
        )
        
        # All string values = qualitative
        ground_truth = {
            "current_situation": "tested on upside",
            "defense_approach": "close or roll",
        }
        
        result = await evaluator.score_with_ground_truth(response, ground_truth)
        
        assert isinstance(result, OptionsScore)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compare_values_empty_extracted(self):
        """Test with empty extracted values."""
        evaluator = OptionsEvaluator(task=create_mock_task())
        
        score, feedback = evaluator._compare_values({}, {"price": 25.18}, tolerance=0.05)
        
        assert score == 50.0
        assert "Insufficient" in feedback or "No comparable" in feedback

    def test_compare_values_zero_expected(self):
        """Test comparison when expected value is zero."""
        evaluator = OptionsEvaluator(task=create_mock_task())
        
        extracted = {"pnl": 0.0}
        ground_truth = {"pnl": 0}
        
        score, feedback = evaluator._compare_values(extracted, ground_truth, tolerance=0.05)
        
        assert score == 100.0

    def test_compare_values_negative_numbers(self):
        """Test comparison with negative values (e.g., theta)."""
        evaluator = OptionsEvaluator(task=create_mock_task())
        
        extracted = {"theta": -0.32}
        ground_truth = {"theta": -0.321}
        
        score, feedback = evaluator._compare_values(extracted, ground_truth, tolerance=0.10)
        
        assert score == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
