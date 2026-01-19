"""
Evaluator Node - Quality assessment of Thai prose
Scores drafts based on KPIs and provides feedback
"""
import os
import json
import re
from typing import Dict, Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from state import KPIReport

load_dotenv()


class EvaluationResult(BaseModel):
    """Structured evaluation result"""
    consistency_score: float = Field(ge=0, le=10)
    consistency_reasoning: str
    prose_quality_score: float = Field(ge=0, le=10)
    prose_quality_reasoning: str
    emotional_score: float = Field(ge=0, le=10)
    emotional_reasoning: str
    overall_feedback: str
    suggestions: list[str] = Field(default_factory=list)


class EvaluatorNode:
    """
    Evaluator Node for AI Novelist

    Responsibilities:
    - Score Thai prose on multiple KPIs
    - Provide detailed feedback for improvement
    - Determine if draft passes quality thresholds
    """

    def __init__(self):
        self.model_name = os.getenv("EVALUATOR_MODEL", "gemma2:9b")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Thresholds
        self.min_consistency = float(os.getenv("MIN_CONSISTENCY_SCORE", "7"))
        self.min_prose_quality = float(os.getenv("MIN_PROSE_QUALITY_SCORE", "7"))
        self.min_emotional = float(os.getenv("MIN_EMOTIONAL_SCORE", "6"))

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=0.3  # Lower for consistent evaluation
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{input}")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _get_system_prompt(self) -> str:
        return """You are an expert Thai literature critic and editor.

Your task is to evaluate Thai prose based on three criteria, scoring each from 0-10.

## Evaluation Criteria

### 1. Consistency (0-10)
- Does the prose follow the scene instructions correctly?
- Are character behaviors consistent with their profiles?
- Is the plot progression logical?
- Are there any contradictions or errors?

### 2. Thai Prose Quality (0-10)
- Is the Thai natural and fluent (not translated-sounding)?
- Are Thai idioms and expressions used appropriately?
- Is the vocabulary rich and varied?
- Do sentences flow well together?
- Are there any grammar or spelling issues?

### 3. Emotional Impact (0-10)
- Does the writing evoke the intended emotions?
- Is the atmosphere effectively created?
- Do readers connect with the characters?
- Is the pacing appropriate for the emotional beats?

## Output Format

You MUST respond in this exact JSON format:
```json
{
    "consistency_score": <number 0-10>,
    "consistency_reasoning": "<explanation>",
    "prose_quality_score": <number 0-10>,
    "prose_quality_reasoning": "<explanation>",
    "emotional_score": <number 0-10>,
    "emotional_reasoning": "<explanation>",
    "overall_feedback": "<summary of main issues and strengths>",
    "suggestions": ["<specific suggestion 1>", "<specific suggestion 2>", ...]
}
```

Be constructive but honest. If the prose has issues, clearly identify them.
Write your reasoning and feedback in Thai to better assess Thai language quality."""

    def evaluate(
        self,
        draft_content: str,
        scene_instructions: str,
        world_context: str = "",
        style: str = "modern_thai"
    ) -> KPIReport:
        """
        Evaluate Thai prose draft

        Args:
            draft_content: The Thai prose to evaluate
            scene_instructions: Original scene instructions
            world_context: Character/world information
            style: Expected writing style

        Returns:
            KPIReport with scores and feedback
        """
        input_text = f"""Please evaluate the following Thai prose draft.

## Writing Style Expected
{style}

## Scene Instructions (What was supposed to be written)
{scene_instructions}

## World Context (Characters & Settings)
{world_context if world_context else "No specific context provided."}

## Thai Prose to Evaluate
---
{draft_content}
---

Evaluate this prose using the three criteria. Respond ONLY with the JSON format specified."""

        result = self.chain.invoke({"input": input_text})

        # Parse the result
        evaluation = self._parse_evaluation(result)

        # Create KPI Report
        kpi = KPIReport(
            consistency_score=evaluation.get("consistency_score", 5.0),
            prose_quality_score=evaluation.get("prose_quality_score", 5.0),
            emotional_score=evaluation.get("emotional_score", 5.0),
            feedback=self._build_feedback(evaluation)
        )

        # Calculate average and check if passed
        kpi.calculate_average()
        kpi.is_passed = self._check_passed(kpi)

        return kpi

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from evaluator"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract scores manually
        return self._extract_scores_manually(response)

    def _extract_scores_manually(self, response: str) -> Dict[str, Any]:
        """Extract scores from non-JSON response"""
        result = {
            "consistency_score": 5.0,
            "consistency_reasoning": "",
            "prose_quality_score": 5.0,
            "prose_quality_reasoning": "",
            "emotional_score": 5.0,
            "emotional_reasoning": "",
            "overall_feedback": response,
            "suggestions": []
        }

        # Try to find scores in text
        patterns = {
            "consistency_score": r"consistency[:\s]*(\d+(?:\.\d+)?)",
            "prose_quality_score": r"(?:prose|quality)[:\s]*(\d+(?:\.\d+)?)",
            "emotional_score": r"emotional[:\s]*(\d+(?:\.\d+)?)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result[key] = min(10.0, float(match.group(1)))

        return result

    def _build_feedback(self, evaluation: Dict[str, Any]) -> str:
        """Build comprehensive feedback string"""
        feedback_parts = []

        if evaluation.get("overall_feedback"):
            feedback_parts.append(f"Overall: {evaluation['overall_feedback']}")

        if evaluation.get("consistency_reasoning"):
            feedback_parts.append(f"Consistency: {evaluation['consistency_reasoning']}")

        if evaluation.get("prose_quality_reasoning"):
            feedback_parts.append(f"Prose Quality: {evaluation['prose_quality_reasoning']}")

        if evaluation.get("emotional_reasoning"):
            feedback_parts.append(f"Emotional Impact: {evaluation['emotional_reasoning']}")

        if evaluation.get("suggestions"):
            suggestions = "\n- ".join(evaluation["suggestions"])
            feedback_parts.append(f"Suggestions:\n- {suggestions}")

        return "\n\n".join(feedback_parts)

    def _check_passed(self, kpi: KPIReport) -> bool:
        """Check if KPI scores meet minimum thresholds"""
        return (
            kpi.consistency_score >= self.min_consistency and
            kpi.prose_quality_score >= self.min_prose_quality and
            kpi.emotional_score >= self.min_emotional
        )

    def quick_check(self, draft_content: str) -> Dict[str, Any]:
        """
        Quick check for basic issues without full evaluation

        Returns dict with basic metrics
        """
        # Check basic metrics
        char_count = len(draft_content)
        has_thai = bool(re.search(r'[\u0E00-\u0E7F]', draft_content))
        english_ratio = len(re.findall(r'[a-zA-Z]', draft_content)) / max(1, char_count)

        return {
            "char_count": char_count,
            "has_thai": has_thai,
            "english_ratio": english_ratio,
            "is_valid": has_thai and english_ratio < 0.1 and char_count > 100
        }


def create_evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for evaluation

    Args:
        state: Current graph state

    Returns:
        Updated state with kpi_report
    """
    evaluator = EvaluatorNode()

    # Quick check first
    quick = evaluator.quick_check(state.get("draft_content", ""))

    if not quick["is_valid"]:
        # Draft is invalid, create failed report
        kpi = KPIReport(
            consistency_score=0,
            prose_quality_score=0,
            emotional_score=0,
            is_passed=False,
            feedback=f"Draft validation failed: Thai chars: {quick['has_thai']}, English ratio: {quick['english_ratio']:.2%}, Length: {quick['char_count']}"
        )
    else:
        # Full evaluation
        kpi = evaluator.evaluate(
            draft_content=state.get("draft_content", ""),
            scene_instructions=state.get("scene_instructions", ""),
            world_context=state.get("retrieved_context", ""),
            style=state.get("style", "modern_thai")
        )

    return {
        **state,
        "kpi_report": kpi.model_dump()
    }
