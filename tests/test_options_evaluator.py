"""
Tests for OptionsEvaluator including module import and regex functionality.
"""
import pytest
from datetime import datetime, timezone

from cio_agent.models import (
    Task,
    TaskCategory,
    TaskDifficulty,
    GroundTruth,
    TaskRubric,
    AgentResponse,
)


class TestOptionsEvaluatorImport:
    """Test that OptionsEvaluator imports correctly and has access to all dependencies."""

    def test_import_options_evaluator(self):
        """Test that OptionsEvaluator can be imported successfully."""
        from evaluators.options import OptionsEvaluator
        assert OptionsEvaluator is not None

    def test_import_re_module_available(self):
        """Test that re module is accessible in options evaluator module."""
        from evaluators import options
        assert hasattr(options, 're')
        assert options.re is not None

    def test_options_evaluator_instantiation(self):
        """Test that OptionsEvaluator can be instantiated."""
        from evaluators.options import OptionsEvaluator

        task = Task(
            question_id="test_001",
            category=TaskCategory.OPTIONS_PRICING,
            difficulty=TaskDifficulty.MEDIUM,
            question="Calculate call option price",
            ticker="SPY",
            fiscal_year=2025,
            simulation_date=datetime.now(timezone.utc),
            ground_truth=GroundTruth(macro_thesis="test", key_themes=[]),
            rubric=TaskRubric(criteria=[], penalty_conditions=[]),
        )

        evaluator = OptionsEvaluator(task=task)
        assert evaluator is not None
        assert evaluator.task == task


class TestOptionsEvaluatorRegex:
    """Test regex extraction methods in OptionsEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create an OptionsEvaluator instance for testing."""
        from evaluators.options import OptionsEvaluator

        task = Task(
            question_id="test_regex",
            category=TaskCategory.GREEKS_ANALYSIS,
            difficulty=TaskDifficulty.MEDIUM,
            question="Analyze option Greeks",
            ticker="SPY",
            fiscal_year=2025,
            simulation_date=datetime.now(timezone.utc),
            ground_truth=GroundTruth(macro_thesis="test", key_themes=[]),
            rubric=TaskRubric(criteria=[], penalty_conditions=[]),
        )
        return OptionsEvaluator(task=task)

    def test_extract_numbers_from_text(self, evaluator):
        """Test _extract_numbers_from_text method."""
        text = "The price is $45.50 and the premium is $2.30"
        numbers = evaluator._extract_numbers_from_text(text)
        assert 45.50 in numbers
        assert 2.30 in numbers

    def test_extract_greek_delta_simple(self, evaluator):
        """Test extracting delta value from simple text."""
        text = "Delta: 0.42"
        delta = evaluator._extract_greek_value(text, "delta")
        assert delta == pytest.approx(0.42)

    def test_extract_greek_delta_equals(self, evaluator):
        """Test extracting delta with equals sign."""
        text = "delta = 0.65"
        delta = evaluator._extract_greek_value(text, "delta")
        assert delta == pytest.approx(0.65)

    def test_extract_greek_gamma(self, evaluator):
        """Test extracting gamma value."""
        text = "Gamma: 0.025"
        gamma = evaluator._extract_greek_value(text, "gamma")
        assert gamma == pytest.approx(0.025)

    def test_extract_greek_theta(self, evaluator):
        """Test extracting negative theta value."""
        text = "Theta: -0.15"
        theta = evaluator._extract_greek_value(text, "theta")
        assert theta == pytest.approx(-0.15)

    def test_extract_greek_vega(self, evaluator):
        """Test extracting vega value."""
        text = "Vega: 0.35"
        vega = evaluator._extract_greek_value(text, "vega")
        assert vega == pytest.approx(0.35)

    def test_extract_greek_bullet_format(self, evaluator):
        """Test extracting from bullet point format."""
        text = "- Delta: 0.48\n- Gamma: 0.03"
        delta = evaluator._extract_greek_value(text, "delta")
        gamma = evaluator._extract_greek_value(text, "gamma")
        assert delta == pytest.approx(0.48)
        assert gamma == pytest.approx(0.03)

    def test_extract_greek_markdown_bold(self, evaluator):
        """Test extracting from markdown bold format."""
        text = "**Delta**: 0.55"
        delta = evaluator._extract_greek_value(text, "delta")
        assert delta == pytest.approx(0.55)

    def test_extract_greek_not_found(self, evaluator):
        """Test that None is returned when Greek not found."""
        text = "No Greeks mentioned here"
        delta = evaluator._extract_greek_value(text, "delta")
        assert delta is None


class TestOptionsEvaluatorScoring:
    """Test scoring functionality of OptionsEvaluator."""

    @pytest.fixture
    def task(self):
        """Create a test task."""
        return Task(
            question_id="test_score",
            category=TaskCategory.OPTIONS_PRICING,
            difficulty=TaskDifficulty.MEDIUM,
            question="Price a call option on SPY with strike $450, 45 DTE, IV=45%",
            ticker="SPY",
            fiscal_year=2025,
            simulation_date=datetime.now(timezone.utc),
            ground_truth=GroundTruth(
                macro_thesis="Expected price: $12.50, Delta: 0.52",
                key_themes=["options pricing"],
            ),
            rubric=TaskRubric(
                criteria=["Price accuracy within 5%", "Greeks accuracy within 10%"],
                penalty_conditions=[],
            ),
        )

    @pytest.mark.asyncio
    async def test_score_with_greeks(self, task):
        """Test scoring with Greeks extracted."""
        from evaluators.options import OptionsEvaluator

        response = AgentResponse(
            agent_id="test_agent",
            task_id="test_score",
            analysis="""
            Based on Black-Scholes model:
            - Call option price: $12.30
            - Delta: 0.51
            - Gamma: 0.04
            - Theta: -0.08
            - Vega: 0.19
            
            The option is slightly out-of-the-money with moderate positive delta.
            """,
            recommendation="BUY",
        )

        evaluator = OptionsEvaluator(task=task)
        score = await evaluator.score(response)

        assert score is not None
        assert 0 <= score.score <= 100
        assert score.greeks_accuracy > 0  # Should extract Greeks
        assert "delta" in score.feedback.lower() or "greeks" in score.feedback.lower()

    @pytest.mark.asyncio
    async def test_score_without_greeks(self, task):
        """Test scoring when no Greeks are mentioned."""
        from evaluators.options import OptionsEvaluator

        response = AgentResponse(
            agent_id="test_agent",
            task_id="test_score",
            analysis="The option price is approximately $12.50",
            recommendation="HOLD",
        )

        evaluator = OptionsEvaluator(task=task)
        score = await evaluator.score(response)

        assert score is not None
        assert 0 <= score.score <= 100
        # Greeks accuracy should be low since none were provided
        assert score.greeks_accuracy < 50


class TestOptionsEvaluatorRiskManagement:
    """Test risk management scoring."""

    @pytest.fixture
    def task(self):
        """Create a risk management task."""
        return Task(
            question_id="test_risk",
            category=TaskCategory.RISK_MANAGEMENT,
            difficulty=TaskDifficulty.HARD,
            question="Design a risk management strategy for an iron condor position",
            ticker="SPY",
            fiscal_year=2025,
            simulation_date=datetime.now(timezone.utc),
            ground_truth=GroundTruth(
                macro_thesis="Comprehensive risk management required",
                key_themes=["position sizing", "exit strategy", "hedging"],
            ),
            rubric=TaskRubric(
                criteria=["Position sizing", "Max loss defined", "Exit strategy"],
                penalty_conditions=[],
            ),
        )

    @pytest.mark.asyncio
    async def test_risk_score_comprehensive(self, task):
        """Test risk scoring with comprehensive analysis."""
        from evaluators.options import OptionsEvaluator

        response = AgentResponse(
            agent_id="test_agent",
            task_id="test_risk",
            analysis="""
            Risk Management Strategy:
            
            1. Position Sizing: Allocate 2% of portfolio per trade
            2. Maximum Loss: $500 per iron condor (width of widest spread)
            3. Exit Strategy: Close position at 50% of max profit or 50% of max loss
            4. Hedging: Use protective puts if underlying moves outside wings
            5. VaR: 95% VaR estimated at $450
            
            Monitor position daily and adjust strikes as needed.
            """,
            recommendation="IMPLEMENT with strict risk limits",
        )

        evaluator = OptionsEvaluator(task=task)
        score = await evaluator.score(response)

        assert score is not None
        assert score.risk_management > 70  # Should score high with all elements present
        # Check that keywords are detected in feedback
        assert any(
            keyword in score.feedback.lower()
            for keyword in ["position", "loss", "exit", "var"]
        )

    @pytest.mark.asyncio
    async def test_risk_score_minimal(self, task):
        """Test risk scoring with minimal analysis."""
        from evaluators.options import OptionsEvaluator

        response = AgentResponse(
            agent_id="test_agent",
            task_id="test_risk",
            analysis="Use an iron condor strategy.",
            recommendation="TRADE",
        )

        evaluator = OptionsEvaluator(task=task)
        score = await evaluator.score(response)

        assert score is not None
        # Should get low risk management score (base score only)
        assert score.risk_management <= 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
