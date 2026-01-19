"""
Planner Node - Scene planning and instruction generation
Processes plot into detailed English scene beats
"""
import os
from typing import Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


class PlannerNode:
    """
    Planner Node for AI Novelist

    Responsibilities:
    - Analyze overall plot and current chapter requirements
    - Generate detailed English scene instructions
    - Identify characters and settings needed for the scene
    """

    def __init__(self):
        self.model_name = os.getenv("PLANNER_MODEL", "llama3:8b")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=0.7
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{input}")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _get_system_prompt(self) -> str:
        return """You are a professional Thai novel plot architect and scene planner.

Your role is to create detailed, actionable scene instructions in English that will guide a creative writer to produce high-quality Thai prose.

You will receive a BLUEPRINT for each episode that contains:
- Scene structure (Scene 1, Scene 2, etc.)
- Characters involved
- Tone and mood
- Selling points (key moments readers want)
- Cliffhanger requirement
- Intimacy level guidelines

Your job is to EXPAND this blueprint into detailed writing instructions.

For each scene in the episode, provide:

1. **Scene Overview**: What happens (2-3 sentences)

2. **Setting Details**:
   - Time of day
   - Location description
   - Atmosphere/mood
   - Sensory details (sights, sounds, smells)

3. **Characters Present**:
   - List all characters
   - Their emotional states
   - Their goals in this scene

4. **Key Dialogue Points**:
   - Important conversations
   - Tone of dialogue
   - Specific revelations or phrases

5. **Action Beats**:
   - Numbered list of events
   - Physical actions
   - Internal thoughts to show

6. **Intimacy Guidelines** (if applicable):
   - Level of physical description allowed
   - What to show vs imply
   - Emotional focus

7. **Emotional Arc**:
   - Reader feeling at start
   - Reader feeling at end
   - Key turning points

8. **Cliffhanger Setup**:
   - How to build tension toward the cliffhanger
   - Exact ending moment

Be specific and detailed. The writer will use these instructions to create prose in Thai."""

    def plan_scene(
        self,
        plot_summary: str,
        chapter_number: int,
        total_chapters: int,
        style: str,
        previous_context: str = "",
        world_context: str = ""
    ) -> str:
        """
        Generate scene instructions for a chapter

        Args:
            plot_summary: Overall plot of the novel
            chapter_number: Current chapter number
            total_chapters: Total planned chapters
            style: Writing style ('ancient_chinese', 'thai_period', 'modern_thai')
            previous_context: Summary of previous chapters
            world_context: Retrieved character/world knowledge

        Returns:
            Detailed English scene instructions
        """
        input_text = f"""Please create detailed scene instructions for Chapter {chapter_number} of {total_chapters}.

## Overall Plot Summary
{plot_summary}

## Writing Style
{self._get_style_description(style)}

## World Context (Characters & Settings)
{world_context if world_context else "No specific world context provided."}

## Previous Chapter Context
{previous_context if previous_context else "This is the first chapter."}

---

Based on the above information, create comprehensive scene instructions for Chapter {chapter_number}.
Consider the pacing: {"This is the opening - establish the world and hook the reader." if chapter_number == 1 else "Build on previous events." if chapter_number < total_chapters else "This is the finale - bring resolution and emotional payoff."}
"""

        result = self.chain.invoke({"input": input_text})
        return result

    def _get_style_description(self, style: str) -> str:
        """Get style-specific instructions for the planner"""
        style_descriptions = {
            "ancient_chinese": """
Style: Ancient Chinese (จีนโบราณ)
- Setting: Ancient Chinese imperial era
- Language: Formal, poetic
- Characters use titles and formal address
- Include Chinese cultural elements (tea ceremonies, martial arts, palace intrigue)
- Reference nature and seasons metaphorically
""",
            "thai_period": """
Style: Thai Period (ไทยยุคเก่า)
- Setting: Historical Thailand (Ayutthaya, Rattanakosin, etc.)
- Language: Elegant, traditional Thai expressions
- Strong hierarchy in social interactions
- Include Thai cultural elements (temples, traditions, court customs)
- Respect for elders and social status is crucial
""",
            "modern_thai": """
Style: Modern Thai (ไทยสมัยใหม่)
- Setting: Contemporary Thailand
- Language: Natural, conversational
- Modern relationships and social dynamics
- Include contemporary Thai life elements
- Balance formal and casual speech appropriately
"""
        }
        return style_descriptions.get(style, style_descriptions["modern_thai"])

    def plan_from_blueprint(
        self,
        blueprint: dict,
        characters: list,
        previous_context: str = "",
        style: str = "modern_thai"
    ) -> str:
        """
        Generate scene instructions from episode blueprint

        Args:
            blueprint: Episode blueprint dict from database
            characters: List of character info dicts
            previous_context: Summary of previous episodes
            style: Writing style

        Returns:
            Detailed English scene instructions
        """
        # Format characters info
        char_info = ""
        if characters:
            char_info = "\n".join([
                f"- {c.get('name', 'Unknown')}: {c.get('description', '')} "
                f"(Role: {c.get('metadata', {}).get('role', 'unknown')})"
                for c in characters
            ])

        # Get intimacy level guidelines
        intimacy_guidelines = self._get_intimacy_guidelines(blueprint.get('intimacy_level', 'L1'))

        input_text = f"""Please create detailed scene instructions for Episode {blueprint.get('episode_number', 1)}.

## Episode Blueprint
**Title:** {blueprint.get('title', 'Untitled')}
**Arc:** {blueprint.get('arc_name', '')} (Arc {blueprint.get('arc_number', 1)})
**Tone:** {blueprint.get('tone', 'Not specified')}
**Selling Points:** {blueprint.get('selling_points', 'Not specified')}
**Cliffhanger:** {blueprint.get('cliffhanger', 'Not specified')}
**Intimacy Level:** {blueprint.get('intimacy_level', 'L1')}

## Scene Structure from Blueprint
{blueprint.get('scene_structure', 'No structure provided')}

## Notes
{blueprint.get('notes', 'No additional notes')}

## Characters in This Episode
{char_info if char_info else 'No specific characters listed'}

## Writing Style
{self._get_style_description(style)}

## Intimacy Level Guidelines
{intimacy_guidelines}

## Previous Episode Context
{previous_context if previous_context else "This is the first episode or no previous context available."}

---

Based on this blueprint, create comprehensive scene-by-scene instructions.
IMPORTANT:
1. Follow the scene structure exactly as given
2. Include the cliffhanger at the end
3. Respect the intimacy level guidelines
4. Capture the specified tone throughout
5. Make sure the selling points are featured prominently
"""

        result = self.chain.invoke({"input": input_text})
        return result

    def _get_intimacy_guidelines(self, level: str) -> str:
        """Get writing guidelines for intimacy level"""
        guidelines = {
            "L1": """Level 1 - Eye contact and conversation only
- Focus on dialogue and emotional tension
- Describe looks, glances, subtle body language
- No physical contact beyond incidental
- Build romantic tension through words and atmosphere""",

            "L2": """Level 2 - Symbolic touch (holding hands, hugging)
- Can include hand-holding, hugs, light touches
- Describe the emotional impact of touches
- Keep physical descriptions tasteful
- Focus on the emotional significance""",

            "L3": """Level 3 - Closer intimacy (kissing, implied sleeping together)
- Can include kissing scenes with moderate detail
- Sleeping together should be IMPLIED, not explicit
- Use phrases like "that night..." or "she didn't go back to her room"
- Focus on emotional aftermath""",

            "L4": """Level 4 - Soft explicit (implied NC with some detail)
- Can describe the lead-up and aftermath
- Use suggestive language but not graphic
- Focus on emotions and sensations over mechanics
- Can mention clothing removal, positions implied
- Cut away at key moments or use metaphorical language""",

            "L5": """Level 5 - Explicit (full NC description)
- Can include detailed intimate scenes
- Describe physical sensations and actions
- Maintain literary quality - not crude
- Balance explicit content with emotional depth
- Include dialogue and emotional reactions"""
        }
        return guidelines.get(level, guidelines["L1"])

    def refine_instructions(
        self,
        original_instructions: str,
        evaluator_feedback: str,
        draft_content: str
    ) -> str:
        """
        Refine scene instructions based on evaluator feedback

        Used when a draft fails evaluation and needs rewriting.
        """
        refine_prompt = f"""The previous draft based on these instructions did not meet quality standards.

## Original Instructions
{original_instructions}

## Draft That Failed
{draft_content[:1000]}...

## Evaluator Feedback
{evaluator_feedback}

---

Please provide REVISED scene instructions that address the feedback.
Focus on:
1. Areas that need improvement based on feedback
2. Specific guidance to avoid the previous issues
3. Any clarifications or additional details needed

Keep the core story the same, but adjust the execution guidance."""

        result = self.chain.invoke({"input": refine_prompt})
        return result


def create_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for planning

    Args:
        state: Current graph state

    Returns:
        Updated state with scene_instructions
    """
    planner = PlannerNode()

    # Get previous context if available
    previous_context = ""
    if state.get("current_chapter", 1) > 1:
        previous_context = f"Previous chapter ended with the story progressing. Current iteration: {state.get('iteration_count', 0)}"

    # Generate instructions
    instructions = planner.plan_scene(
        plot_summary=state.get("plot_summary", ""),
        chapter_number=state.get("current_chapter", 1),
        total_chapters=state.get("total_chapters", 1),
        style=state.get("style", "modern_thai"),
        previous_context=previous_context,
        world_context=state.get("retrieved_context", "")
    )

    return {
        **state,
        "scene_instructions": instructions
    }
