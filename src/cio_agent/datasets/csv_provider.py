"""
CSV-backed dataset provider for finance Q/A (e.g., data/public.csv).

Expected columns:
    Question, Answer, Question Type, Expert time (mins), Rubric
"""

import csv
import json
from pathlib import Path
from typing import List

from cio_agent.datasets.base import DatasetExample, DatasetProvider
from cio_agent.models import (
    FABQuestionTemplate,
    GroundTruth,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
)

# Map CSV "Question Type" to internal categories
QUESTION_TYPE_MAP = {
    "Quantitative Retrieval": TaskCategory.QUANTITATIVE_RETRIEVAL,
    "Qualitative Retrieval": TaskCategory.QUALITATIVE_RETRIEVAL,
    "Numerical Reasoning": TaskCategory.NUMERICAL_REASONING,
    "Complex Retrieval": TaskCategory.COMPLEX_RETRIEVAL,
    "Adjustments": TaskCategory.ADJUSTMENTS,
    "Beat or Miss": TaskCategory.BEAT_OR_MISS,
    "Trends": TaskCategory.TRENDS,
    "Financial Modeling": TaskCategory.FINANCIAL_MODELING,
    "Market Analysis": TaskCategory.MARKET_ANALYSIS,
}


def _map_difficulty(expert_minutes: float) -> TaskDifficulty:
    """Heuristic mapping from expert time to difficulty."""
    if expert_minutes <= 5:
        return TaskDifficulty.EASY
    if expert_minutes <= 15:
        return TaskDifficulty.MEDIUM
    if expert_minutes <= 30:
        return TaskDifficulty.HARD
    return TaskDifficulty.EXPERT


def _parse_rubric(raw: str) -> tuple[list[str], list[str]]:
    """
    Parse rubric JSON (list of {operator, criteria}) into criteria and penalties.
    Falls back to a simple single-criteria list if parsing fails.
    """
    if not raw:
        return [], []

    try:
        data = json.loads(raw)
        criteria, penalties = [], []
        for item in data:
            op = item.get("operator")
            crit = item.get("criteria")
            if not crit:
                continue
            if op == "correctness":
                criteria.append(crit)
            elif op == "contradiction":
                penalties.append(crit)
        return criteria, penalties
    except Exception:
        return [raw], []


class CsvFinanceDatasetProvider(DatasetProvider):
    """Provider that reads finance Q/A rows from a CSV file."""

    name = "csv_finance"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> List[DatasetExample]:
        rows: List[DatasetExample] = []
        with self.path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                category = QUESTION_TYPE_MAP.get(
                    row.get("Question Type", ""), TaskCategory.QUALITATIVE_RETRIEVAL
                )
                difficulty = _map_difficulty(float(row.get("Expert time (mins)") or 10))
                criteria, penalties = _parse_rubric(row.get("Rubric", ""))
                rubric = TaskRubric(criteria=criteria, penalty_conditions=penalties)

                ground_truth = GroundTruth(
                    macro_thesis=row.get("Answer", ""),
                    key_themes=criteria,
                )

                rows.append(
                    DatasetExample(
                        example_id=f"{self.name}_{idx}",
                        question=row.get("Question", ""),
                        answer=row.get("Answer"),
                        rubric=rubric,
                        category=category,
                        difficulty=difficulty,
                        ground_truth=ground_truth,
                        source=self.name,
                        metadata={"row_index": idx, "raw": row},
                    )
                )
        return rows

    def to_templates(self) -> List[FABQuestionTemplate]:
        """Convert CSV rows directly into FAB templates."""
        templates: List[FABQuestionTemplate] = []
        for ex in self.load():
            templates.append(
                FABQuestionTemplate(
                    template_id=ex.example_id,
                    category=ex.category,
                    template=ex.question,
                    difficulty=ex.difficulty,
                    metric="custom",
                    rubric=ex.rubric or TaskRubric(criteria=[]),
                    requires_code_execution=False,
                )
            )
        return templates
