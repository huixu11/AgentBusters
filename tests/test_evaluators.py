"""
Unit tests for dataset-specific evaluators.
"""

import pytest
from evaluators.base import EvalResult
from evaluators.bizfinbench_evaluator import BizFinBenchEvaluator
from evaluators.public_csv_evaluator import PublicCsvEvaluator


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_percentage(self):
        result = EvalResult(score=0.8, max_score=1.0)
        assert result.percentage == 80.0

    def test_is_correct_true(self):
        result = EvalResult(score=1.0, max_score=1.0)
        assert result.is_correct is True

    def test_is_correct_false(self):
        result = EvalResult(score=0.5, max_score=1.0)
        assert result.is_correct is False


class TestBizFinBenchEvaluator:
    """Test BizFinBenchEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return BizFinBenchEvaluator()

    # Numerical tests
    def test_numerical_exact_match(self, evaluator):
        result = evaluator.evaluate("1.2532", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 1.0
        assert result.is_correct

    def test_numerical_within_tolerance(self, evaluator):
        # 1% tolerance: 1.2532 * 0.01 = 0.0125, so 1.26 should pass
        result = evaluator.evaluate("1.26", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 1.0

    def test_numerical_outside_tolerance(self, evaluator):
        # 1.30 is > 1% away from 1.2532
        result = evaluator.evaluate("1.30", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 0.0

    def test_numerical_with_text(self, evaluator):
        result = evaluator.evaluate("The answer is 1.2532", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 1.0

    # Sequence tests
    def test_sequence_exact_match(self, evaluator):
        result = evaluator.evaluate("2,1,4,3", "2,1,4,3", task_type="event_logic_reasoning")
        assert result.score == 1.0
        assert result.is_correct

    def test_sequence_wrong_order(self, evaluator):
        result = evaluator.evaluate("1,2,3,4", "2,1,4,3", task_type="event_logic_reasoning")
        assert result.score == 0.0

    def test_sequence_with_spaces(self, evaluator):
        result = evaluator.evaluate("2, 1, 4, 3", "2,1,4,3", task_type="event_logic_reasoning")
        assert result.score == 1.0

    # Classification tests
    def test_classification_match(self, evaluator):
        result = evaluator.evaluate("positive", "positive", task_type="user_sentiment_analysis")
        assert result.score == 1.0

    def test_classification_case_insensitive(self, evaluator):
        result = evaluator.evaluate("POSITIVE", "positive", task_type="user_sentiment_analysis")
        assert result.score == 1.0

    def test_classification_mismatch(self, evaluator):
        result = evaluator.evaluate("negative", "positive", task_type="user_sentiment_analysis")
        assert result.score == 0.0

    # Edge cases
    def test_empty_prediction(self, evaluator):
        result = evaluator.evaluate("", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 0.0

    def test_empty_expected(self, evaluator):
        result = evaluator.evaluate("1.2532", "", task_type="financial_quantitative_computation")
        assert result.score == 0.0


class TestPublicCsvEvaluator:
    """Test PublicCsvEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return PublicCsvEvaluator()

    def test_all_correctness_met(self, evaluator):
        rubric = [
            {"operator": "correctness", "criteria": "Q3 2024"},
            {"operator": "correctness", "criteria": "revenue growth"},
        ]
        result = evaluator.evaluate("In Q3 2024, revenue growth was 15%", rubric=rubric)
        assert result.score == 1.0

    def test_partial_correctness(self, evaluator):
        rubric = [
            {"operator": "correctness", "criteria": "Q3 2024"},
            {"operator": "correctness", "criteria": "revenue growth"},
        ]
        result = evaluator.evaluate("Q3 2024 results", rubric=rubric)
        # Should get partial credit (1 out of 2)
        assert 0 < result.score < 1.0

    def test_no_correctness_met(self, evaluator):
        rubric = [
            {"operator": "correctness", "criteria": "Q3 2024"},
            {"operator": "correctness", "criteria": "revenue growth"},
        ]
        result = evaluator.evaluate("Something unrelated", rubric=rubric)
        assert result.score == 0.0

    def test_empty_rubric(self, evaluator):
        result = evaluator.evaluate("Some answer", "Expected", rubric=None)
        # Falls back to simple comparison
        assert result.score >= 0.0

    def test_numerical_criteria(self, evaluator):
        rubric = [
            {"operator": "correctness", "criteria": "$3.25 Billion"},
        ]
        result = evaluator.evaluate("The cost was $3.25 Billion", rubric=rubric)
        assert result.score == 1.0


class TestBizFinBenchEvaluatorBatch:
    """Test batch evaluation."""

    def test_batch_evaluate(self):
        evaluator = BizFinBenchEvaluator()
        predictions = ["1.0", "2.0", "3.0"]
        expected = ["1.0", "2.0", "4.0"]
        
        results = evaluator.evaluate_batch(
            predictions, expected,
            task_type="financial_quantitative_computation"
        )
        
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].score == 1.0
        assert results[2].score == 0.0

    def test_aggregate_results(self):
        evaluator = BizFinBenchEvaluator()
        results = [
            EvalResult(score=1.0, max_score=1.0),
            EvalResult(score=1.0, max_score=1.0),
            EvalResult(score=0.0, max_score=1.0),
        ]
        
        agg = evaluator.aggregate_results(results)
        
        assert agg["count"] == 3
        assert agg["correct_count"] == 2
        assert agg["accuracy"] == 2/3


class TestBizFinBenchEvaluatorEdgeCases:
    """Test BizFinBench error cases."""

    @pytest.fixture
    def evaluator(self):
        return BizFinBenchEvaluator()

    def test_no_extractable_number(self, evaluator):
        """Test prediction with no extractable number."""
        result = evaluator.evaluate("no numbers here", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 0.0
        assert "Could not extract" in result.feedback

    def test_malformed_sequence(self, evaluator):
        """Test malformed sequence input."""
        result = evaluator.evaluate("abc", "2,1,4,3", task_type="event_logic_reasoning")
        assert result.score == 0.0

    def test_whitespace_only_prediction(self, evaluator):
        """Test whitespace-only prediction."""
        result = evaluator.evaluate("   ", "answer", task_type="user_sentiment_analysis")
        assert result.score == 0.0


class TestPublicCsvEvaluatorKeyElements:
    """Test _extract_key_elements method."""

    @pytest.fixture
    def evaluator(self):
        return PublicCsvEvaluator()

    def test_extract_numbers(self, evaluator):
        elements = evaluator._extract_key_elements("Revenue was $3.25 billion in Q4")
        assert "$3.25 billion" in elements or "3.25" in [e for e in elements if "3.25" in e]

    def test_extract_names(self, evaluator):
        elements = evaluator._extract_key_elements("John Smith is the CEO")
        assert "john smith" in elements or any("john" in e for e in elements)

    def test_extract_tickers(self, evaluator):
        elements = evaluator._extract_key_elements("AAPL and MSFT are tech stocks")
        assert "aapl" in elements or "msft" in elements

    def test_empty_text(self, evaluator):
        elements = evaluator._extract_key_elements("")
        assert elements == []

