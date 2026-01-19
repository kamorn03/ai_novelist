"""
Writer Node - Thai prose generation
Converts English scene instructions into high-quality Thai prose
"""
import os
from typing import Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from state import STYLE_GUIDES

load_dotenv()


class WriterNode:
    """
    Writer Node for AI Novelist

    Responsibilities:
    - Convert English scene instructions to Thai prose
    - Apply style-specific writing conventions
    - Use appropriate pronouns and literary devices
    - Ensure natural, non-translated Thai prose
    """

    def __init__(self):
        self.model_name = os.getenv("WRITER_MODEL", "llama3:8b")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Use uncensored model for creative freedom
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=0.8,  # Higher for creativity
            num_predict=4096  # Allow longer outputs
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{input}")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _get_system_prompt(self, style: str) -> str:
        """Generate style-specific system prompt"""
        style_info = STYLE_GUIDES.get(style, STYLE_GUIDES["modern_thai"])

        return f"""You are a master Thai novelist specializing in {style_info['name']} style writing.

## CRITICAL INSTRUCTIONS

1. **Language**: Your output MUST be in Thai only. No English in the final prose.

2. **DO NOT TRANSLATE**: You are not translating from English. You are CREATING original Thai literature.
   - Bad: Direct translation sounds unnatural
   - Good: Natural Thai prose that flows beautifully

3. **Literary Quality**:
   - Use Thai metaphors and idioms
   - Employ varied sentence structures
   - Create vivid sensory descriptions
   - Show emotions through actions, not just statements

4. **Pronoun System**:
{self._format_pronouns(style_info.get('pronouns', {}))}

5. **Style Guidelines**:
{style_info.get('instructions', '')}

## WRITING PROCESS

Think and plan in English internally, then produce ONLY Thai prose as output.

Your response should be:
- Pure Thai prose (no English, no meta-commentary)
- At least 500 Thai characters
- Rich in description and emotion
- True to the style conventions"""

    def _format_pronouns(self, pronouns: Dict) -> str:
        """Format pronouns for the prompt"""
        if not pronouns:
            return "Use appropriate Thai pronouns based on character relationships."

        lines = []
        for category, pronoun_set in pronouns.items():
            pronoun_str = ", ".join([f"{k}: {v}" for k, v in pronoun_set.items()])
            lines.append(f"   - {category}: {pronoun_str}")
        return "\n".join(lines)

    def write_scene(
        self,
        scene_instructions: str,
        style: str,
        world_context: str = "",
        previous_feedback: str = ""
    ) -> str:
        """
        Generate Thai prose from scene instructions

        Args:
            scene_instructions: Detailed English scene beats from Planner
            style: Writing style
            world_context: Character/world information from RAG
            previous_feedback: Feedback from failed evaluation (for rewrites)

        Returns:
            Thai prose
        """
        system_prompt = self._get_system_prompt(style)

        input_text = f"""## Scene Instructions (English - for your understanding)
{scene_instructions}

## World Context (Characters & Settings)
{world_context if world_context else "Create appropriate details as needed."}
"""

        if previous_feedback:
            input_text += f"""
## Previous Attempt Feedback (Address these issues)
{previous_feedback}

Please rewrite the scene addressing the feedback above.
"""

        input_text += """
---

Now write this scene in beautiful, natural Thai prose.
Remember: Output ONLY Thai text. No English. No explanations."""

        result = self.chain.invoke({
            "system_prompt": system_prompt,
            "input": input_text
        })

        return self._clean_output(result)

    def _clean_output(self, text: str) -> str:
        """Clean the output to ensure pure Thai prose"""
        # Remove any potential meta-commentary or English remnants
        lines = text.strip().split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip lines that look like meta-commentary
            if line.strip().startswith(('Note:', 'Translation:', '---', '##', '*')):
                continue
            # Skip empty lines at the start
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def continue_scene(
        self,
        previous_content: str,
        continuation_instructions: str,
        style: str
    ) -> str:
        """
        Continue a scene that needs more content

        Args:
            previous_content: The prose written so far
            continuation_instructions: Instructions for what to add
            style: Writing style

        Returns:
            Continuation in Thai
        """
        system_prompt = self._get_system_prompt(style)

        input_text = f"""## Previous Content (Continue from here)
{previous_content[-1000:]}  # Last 1000 chars for context

## What to Add Next
{continuation_instructions}

---

Continue the story naturally in Thai. Match the tone and style of the previous content.
Output ONLY Thai text."""

        result = self.chain.invoke({
            "system_prompt": system_prompt,
            "input": input_text
        })

        return self._clean_output(result)


def create_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for writing

    Args:
        state: Current graph state

    Returns:
        Updated state with draft_content
    """
    writer = WriterNode()

    # Get feedback from previous iteration if exists
    previous_feedback = ""
    if state.get("kpi_report") and state.get("iteration_count", 0) > 0:
        kpi = state["kpi_report"]
        if isinstance(kpi, dict):
            previous_feedback = kpi.get("feedback", "")

    # Generate Thai prose
    draft = writer.write_scene(
        scene_instructions=state.get("scene_instructions", ""),
        style=state.get("style", "modern_thai"),
        world_context=state.get("retrieved_context", ""),
        previous_feedback=previous_feedback
    )

    # Increment iteration count
    new_iteration = state.get("iteration_count", 0) + 1

    return {
        **state,
        "draft_content": draft,
        "iteration_count": new_iteration
    }
