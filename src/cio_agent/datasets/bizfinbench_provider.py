"""
BizFinBench.v2 dataset provider.

Provides access to the HiThink Research BizFinBench.v2 benchmark dataset,
which contains 29,578 Q&A pairs across 9 financial task types in both
Chinese and English.

Usage:
    provider = BizFinBenchProvider(
        base_path="data/BizFinBench.v2",
        task_type="financial_quantitative_computation",
        language="en",
        limit=100
    )
    examples = provider.load()
    templates = provider.to_templates()
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from cio_agent.datasets.base_jsonl_provider import BaseJSONLProvider
from cio_agent.models import TaskCategory, TaskDifficulty

logger = logging.getLogger(__name__)


class BizFinBenchProvider(BaseJSONLProvider):
    """
    Provider for HiThink BizFinBench.v2 dataset.
    
    Supports all 9 task types:
        - anomaly_information_tracing
        - conterfactual
        - event_logic_reasoning
        - financial_data_description
        - financial_multi_turn_perception
        - financial_quantitative_computation
        - financial_report_analysis (cn only)
        - stock_price_predict
        - user_sentiment_analysis
    """

    name = "bizfinbench"

    # Task type to filename pattern mapping
    TASK_FILES = {
        "anomaly_information_tracing": "anomaly_information_tracing_{lang}.jsonl",
        "conterfactual": "conterfactual_{lang}.jsonl",
        "event_logic_reasoning": "event_logic_reasoning_{lang}.jsonl",
        "financial_data_description": "financial_data_description_{lang}.jsonl",
        "financial_multi_turn_perception": "financial_multi-turn_perception_{lang}.jsonl",
        "financial_quantitative_computation": "financial_quantitative_computation_{lang}.jsonl",
        "financial_report_analysis": "financial_report_analysis.jsonl",  # cn only
        "stock_price_predict": "stock_price_predict_{lang}.jsonl",
        "user_sentiment_analysis": "user_sentiment_analysis_{lang}.jsonl",
    }

    # Map BizFinBench tasks to existing TaskCategory
    TASK_CATEGORY_MAP = {
        "financial_quantitative_computation": TaskCategory.NUMERICAL_REASONING,
        "event_logic_reasoning": TaskCategory.QUALITATIVE_RETRIEVAL,
        "conterfactual": TaskCategory.FINANCIAL_MODELING,
        "stock_price_predict": TaskCategory.MARKET_ANALYSIS,
        "financial_data_description": TaskCategory.QUANTITATIVE_RETRIEVAL,
        "user_sentiment_analysis": TaskCategory.QUALITATIVE_RETRIEVAL,
        "financial_multi_turn_perception": TaskCategory.COMPLEX_RETRIEVAL,
        "anomaly_information_tracing": TaskCategory.COMPLEX_RETRIEVAL,
        "financial_report_analysis": TaskCategory.QUALITATIVE_RETRIEVAL,
    }

    # Difficulty mapping based on task complexity
    TASK_DIFFICULTY_MAP = {
        "financial_quantitative_computation": TaskDifficulty.MEDIUM,
        "event_logic_reasoning": TaskDifficulty.MEDIUM,
        "conterfactual": TaskDifficulty.HARD,
        "stock_price_predict": TaskDifficulty.EXPERT,
        "financial_data_description": TaskDifficulty.EASY,
        "user_sentiment_analysis": TaskDifficulty.MEDIUM,
        "financial_multi_turn_perception": TaskDifficulty.HARD,
        "anomaly_information_tracing": TaskDifficulty.HARD,
        "financial_report_analysis": TaskDifficulty.HARD,
    }

    def __init__(
        self,
        base_path: Union[str, Path],
        task_type: str,
        language: str = "en",
        limit: Optional[int] = None,
    ):
        """
        Initialize BizFinBench provider.
        
        Args:
            base_path: Path to BizFinBench.v2 directory (e.g., "data/BizFinBench.v2")
            task_type: One of the 9 task types (e.g., "financial_quantitative_computation")
            language: "en" for English or "cn" for Chinese (default: "en")
            limit: Optional limit on number of examples to load
            
        Raises:
            ValueError: If task_type is unknown or language is invalid
            FileNotFoundError: If the resolved file path doesn't exist
        """
        self.base_path = Path(base_path)
        self.task_type = task_type
        self.language = language

        # Validate task type
        if task_type not in self.TASK_FILES:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Valid types: {list(self.TASK_FILES.keys())}"
            )

        # Validate language
        if language not in ("en", "cn"):
            raise ValueError(f"Invalid language: {language}. Must be 'en' or 'cn'")

        # Special handling for financial_report_analysis (cn only)
        if task_type == "financial_report_analysis" and language != "cn":
            logger.warning(
                f"Task '{task_type}' is only available in Chinese. "
                f"Switching to language='cn'."
            )
            self.language = "cn"

        # Resolve file path
        file_pattern = self.TASK_FILES[task_type]
        if "{lang}" in file_pattern:
            filename = file_pattern.format(lang=self.language)
        else:
            filename = file_pattern

        file_path = self.base_path / self.language / filename
        
        # Update provider name to be unique per task
        self.name = f"bizfinbench_{task_type}_{self.language}"

        super().__init__(file_path, limit)
        
        logger.info(
            f"Initialized BizFinBenchProvider: task={task_type}, "
            f"lang={self.language}, path={file_path}"
        )

    @classmethod
    def list_task_types(cls, language: str = None) -> List[str]:
        """
        Return list of available task types.
        
        Args:
            language: If specified ('en' or 'cn'), only return tasks available for that language.
                      If None, return all task types.
        
        Returns:
            List of task type names
        """
        all_tasks = list(cls.TASK_FILES.keys())
        
        if language is None:
            return all_tasks
        
        if language == "en":
            # English doesn't have financial_report_analysis
            return [t for t in all_tasks if t != "financial_report_analysis"]
        elif language == "cn":
            # Chinese has all tasks
            return all_tasks
        else:
            raise ValueError(f"Invalid language: {language}. Must be 'en' or 'cn'")
    
    @classmethod
    def list_task_types_by_language(cls) -> dict:
        """
        Return dict of available task types per language.
        
        Returns:
            Dict with 'en' and 'cn' keys, each containing list of available task types
        """
        return {
            "en": cls.list_task_types("en"),
            "cn": cls.list_task_types("cn"),
        }

    def _extract_question(self, item: dict) -> str:
        """
        Extract question from BizFinBench JSONL item.
        
        BizFinBench format:
        {
            "messages": [
                {"role": "user", "content": [{"text": "...", "type": "text"}]}
            ],
            "choices": [...]
        }
        """
        try:
            messages = item.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                return c.get("text", "")
                            elif isinstance(c, str):
                                return c
                    elif isinstance(content, str):
                        return content
        except Exception as e:
            logger.warning(f"Failed to extract question: {e}")
        
        return ""

    def _extract_answer(self, item: dict) -> str:
        """
        Extract answer from BizFinBench JSONL item.
        
        BizFinBench format:
        {
            "messages": [...],
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": [{"text": "...", "type": "text"}]}}
            ]
        }
        """
        try:
            choices = item.get("choices", [])
            for choice in choices:
                msg = choice.get("message", {})
                if msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                return c.get("text", "")
                            elif isinstance(c, str):
                                return c
                    elif isinstance(content, str):
                        return content
        except Exception as e:
            logger.warning(f"Failed to extract answer: {e}")
        
        return ""

    def _get_category(self, item: dict) -> TaskCategory:
        """Get task category based on task type."""
        return self.TASK_CATEGORY_MAP.get(
            self.task_type, TaskCategory.QUALITATIVE_RETRIEVAL
        )

    def _get_difficulty(self, item: dict) -> TaskDifficulty:
        """Get task difficulty based on task type."""
        return self.TASK_DIFFICULTY_MAP.get(
            self.task_type, TaskDifficulty.MEDIUM
        )
