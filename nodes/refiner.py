"""
Refiner Node - Local AI refinement based on Claude's instructions
Applies refinement instructions from Claude to improve Thai prose
"""
import os
from typing import Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from state import STYLE_GUIDES

load_dotenv()


class RefinerNode:
    """
    Refiner Node for AI Novelist

    Responsibilities:
    - Take Claude's draft and refinement instructions
    - Apply improvements using local Ollama model
    - Maintain the original quality while addressing specific issues
    """

    def __init__(self):
        self.model_name = os.getenv("REFINER_MODEL", "llama3:8b")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=0.6,  # Lower than writer for more focused refinement
            num_predict=4096
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{input}")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _get_system_prompt(self, style: str) -> str:
        """Generate system prompt for refinement"""
        style_info = STYLE_GUIDES.get(style, STYLE_GUIDES["modern_thai"])

        return f"""You are a skilled Thai editor refining prose in {style_info['name']} style.

## YOUR ROLE
You are a REFINEMENT SPECIALIST. Your job is to:
1. Take an existing Thai prose draft
2. Apply specific refinement instructions
3. Output an IMPROVED version of the prose

## CRITICAL RULES

1. **Output ONLY Thai prose** - No English, no explanations, no meta-commentary
2. **Preserve the original structure** - Don't completely rewrite, just refine
3. **Focus on the instructions** - Address specifically what's asked
4. **Maintain word count** - The refined version should be similar length or longer
5. **Keep the story** - Don't change plot points or character actions

## STYLE GUIDELINES
{style_info.get('instructions', '')}

## PRONOUN USAGE
{self._format_pronouns(style_info.get('pronouns', {}))}

Your output must be PURE THAI PROSE only."""

    def _format_pronouns(self, pronouns: Dict) -> str:
        """Format pronouns for the prompt"""
        if not pronouns:
            return "Use appropriate Thai pronouns based on character relationships."

        lines = []
        for category, pronoun_set in pronouns.items():
            pronoun_str = ", ".join([f"{k}: {v}" for k, v in pronoun_set.items()])
            lines.append(f"   - {category}: {pronoun_str}")
        return "\n".join(lines)

    def refine_prose(
        self,
        original_draft: str,
        refinement_instructions: Dict[str, Any],
        style: str,
        world_context: str = ""
    ) -> str:
        """
        Refine Thai prose based on Claude's instructions

        Args:
            original_draft: The Thai prose to refine
            refinement_instructions: Instructions from Claude
            style: Writing style
            world_context: Character/world information

        Returns:
            Refined Thai prose
        """
        system_prompt = self._get_system_prompt(style)

        # Format refinement instructions
        instructions_text = self._format_instructions(refinement_instructions)

        input_text = f"""## Original Thai Prose to Refine
---
{original_draft}
---

## Refinement Instructions (from senior editor)
{instructions_text}

## World Context (for reference)
{world_context if world_context else "Use the original context."}

---

Now refine the prose above by applying the instructions.
Output ONLY the refined Thai prose. No explanations."""

        result = self.chain.invoke({
            "system_prompt": system_prompt,
            "input": input_text
        })

        return self._clean_output(result, original_draft)

    def _format_instructions(self, instructions: Dict[str, Any]) -> str:
        """Format refinement instructions for the prompt"""
        if not instructions:
            return "No specific instructions. Just improve overall quality."

        parts = []

        if instructions.get("focus_areas"):
            parts.append("### Focus Areas (Most Important)")
            for i, area in enumerate(instructions["focus_areas"], 1):
                parts.append(f"   {i}. {area}")

        if instructions.get("style_adjustments"):
            parts.append("\n### Style Adjustments")
            for adj in instructions["style_adjustments"]:
                parts.append(f"   - {adj}")

        if instructions.get("specific_fixes"):
            parts.append("\n### Specific Fixes Required")
            for fix in instructions["specific_fixes"]:
                parts.append(f"   - {fix}")

        if instructions.get("enhancement_suggestions"):
            parts.append("\n### Optional Enhancements")
            for sug in instructions["enhancement_suggestions"]:
                parts.append(f"   - {sug}")

        return "\n".join(parts) if parts else "Improve overall prose quality."

    def _clean_output(self, text: str, original: str) -> str:
        """Clean the output and ensure it's valid Thai prose"""
        lines = text.strip().split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip lines that look like meta-commentary
            if line.strip().startswith(('Note:', 'Translation:', '---', '##', '*', '```')):
                continue
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines).strip()

        # If result is too short or empty, return original
        if len(result) < len(original) * 0.5:
            return original

        return result


def create_refiner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for refinement

    Args:
        state: Current graph state

    Returns:
        Updated state with refined draft_content
    """
    # Skip refinement if no instructions
    if not state.get("refinement_instructions"):
        return state

    refiner = RefinerNode()

    # Use claude_draft as source, or draft_content if not available
    original_draft = state.get("claude_draft") or state.get("draft_content", "")

    if not original_draft:
        return state

    # Refine the prose
    refined_content = refiner.refine_prose(
        original_draft=original_draft,
        refinement_instructions=state.get("refinement_instructions", {}),
        style=state.get("style", "modern_thai"),
        world_context=state.get("retrieved_context", "")
    )

    return {
        **state,
        "draft_content": refined_content
    }
