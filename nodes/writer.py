"""
Writer Node - Thai prose generation
Converts English scene instructions into high-quality Thai prose
Supports both Claude API (primary) and Ollama (fallback/refinement)
"""
import os
import json
import re
from typing import Dict, Any, Optional, Tuple, List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Claude integration
try:
    from langchain_anthropic import ChatAnthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

from state import STYLE_GUIDES, RefinementInstructions

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
            num_predict=8192  # Allow longer outputs (for 6500+ Thai chars)
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{input}")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _get_system_prompt(self, style: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """Generate style-specific system prompt with optional few-shot examples"""
        style_info = STYLE_GUIDES.get(style, STYLE_GUIDES["modern_thai"])

        base_prompt = f"""You are a master Thai novelist specializing in {style_info['name']} style writing.

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
{style_info.get('instructions', '')}"""

        # Add few-shot examples if provided
        if few_shot_examples:
            base_prompt += "\n\n## HIGH-QUALITY EXAMPLES\n"
            base_prompt += "Study these examples of excellent Thai prose in this style:\n\n"

            for i, example in enumerate(few_shot_examples, 1):
                # Truncate examples to keep prompt manageable
                input_preview = example['input_context'][:300]
                output_preview = example['output_example'][:500]

                base_prompt += f"### Example {i}\n"
                base_prompt += f"**Input Instructions:**\n{input_preview}...\n\n"
                base_prompt += f"**High-Quality Output:**\n{output_preview}...\n\n"
                base_prompt += f"(Quality Score: {example['quality_score']:.1f}/10)\n\n"

            base_prompt += "Your output should match or exceed this quality level.\n"

        base_prompt += """
## WRITING PROCESS

Think and plan in English internally, then produce ONLY Thai prose as output.

Your response should be:
- Pure Thai prose (no English, no meta-commentary)
- At least 6500 Thai characters (this is a full novel episode, not a short scene)
- Rich in description and emotion
- True to the style conventions
- Detailed dialogue and character interactions
- Multiple scene transitions within the episode"""

        return base_prompt

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
        previous_feedback: str = "",
        project_id: Optional[int] = None,
        use_few_shot: bool = True
    ) -> str:
        """
        Generate Thai prose from scene instructions with optional few-shot learning

        Args:
            scene_instructions: Detailed English scene beats from Planner
            style: Writing style
            world_context: Character/world information from RAG
            previous_feedback: Feedback from failed evaluation (for rewrites)
            project_id: Project ID for fetching relevant examples
            use_few_shot: Whether to use few-shot examples (default: True)

        Returns:
            Thai prose
        """
        # Fetch few-shot examples if enabled
        few_shot_examples = []
        if use_few_shot and project_id and os.getenv("FEW_SHOT_ENABLED", "true").lower() == "true":
            try:
                from nodes.database import DatabaseNode
                db = DatabaseNode()
                few_shot_examples = db.get_few_shot_examples(
                    example_type='writer',
                    style=style,
                    query_context=scene_instructions,
                    limit=int(os.getenv("FEW_SHOT_MAX_EXAMPLES_PER_PROMPT", "2")),
                    min_quality=float(os.getenv("FEW_SHOT_MIN_QUALITY", "8.0"))
                )
                if few_shot_examples:
                    print(f"[Writer] Using {len(few_shot_examples)} few-shot examples")
            except Exception as e:
                print(f"[Writer] Failed to fetch few-shot examples: {e}")

        system_prompt = self._get_system_prompt(style, few_shot_examples)

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


class ClaudeWriterNode:
    """
    Claude-powered Writer Node for AI Novelist

    Responsibilities:
    - Generate high-quality Thai prose using Claude API
    - Provide refinement instructions for local AI to improve
    - Output structured response with prose + instructions
    """

    def __init__(self):
        if not CLAUDE_AVAILABLE:
            raise ImportError("langchain-anthropic is not installed. Run: pip install langchain-anthropic")

        self.api_key = os.getenv("CLAUDE_API_KEY")
        self.model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY not found in environment variables")

        self.llm = ChatAnthropic(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.8,
            max_tokens=4096
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{input}")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _get_system_prompt(self, style: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """Generate style-specific system prompt for Claude with optional few-shot examples"""
        style_info = STYLE_GUIDES.get(style, STYLE_GUIDES["modern_thai"])

        base_prompt = f"""คุณคือนักเขียนนิยายไทยมืออาชีพระดับรางวัลซีไรต์ เชี่ยวชาญการเขียนแนว {style_info['name']}

## บทบาทของคุณ
คุณจะเขียนฉากนิยายภาษาไทยที่มีคุณภาพสูง อ่านสนุก มีอารมณ์ และดึงดูดผู้อ่าน

## เทคนิคการเขียนที่ต้องใช้

### 1. SHOW DON'T TELL (แสดง อย่าบอก)
❌ ไม่ดี: "เธอรู้สึกเศร้ามาก"
✅ ดี: "น้ำตาไหลพรากลงมาโดยไม่รู้ตัว เธอกัดริมฝีปากแน่นจนซีด"

❌ ไม่ดี: "เขาโกรธมาก"
✅ ดี: "กำปั้นของเขาขาวซีดจากการกำแน่น เส้นเลือดบนขมับปูดโปน"

### 2. โครงสร้างฉาก (Scene Structure)
ทุกฉากต้องมี:
- **เป้าหมาย**: ตัวละครต้องการอะไร?
- **อุปสรรค**: อะไรขัดขวาง?
- **ผลลัพธ์**: เกิดอะไรขึ้น? (ควรจบด้วยความขัดแย้งหรือคำถามใหม่)

### 3. บทสนทนาที่มีชีวิต
- ทุกตัวละครต้องมี "เสียง" เฉพาะตัว (word choice, sentence length, attitude)
- บทสนทนาต้องเผยนิสัย ไม่ใช่แค่ให้ข้อมูล
- ใช้ subtext - สิ่งที่ไม่ได้พูดสำคัญเท่าสิ่งที่พูด
- หลีกเลี่ยงบทสนทนาที่ "on the nose" (พูดตรงเกินไป)

### 4. รายละเอียดประสาทสัมผัส (Sensory Details)
ใช้ทั้ง 5 สัมผัส:
- ภาพ: แสง สี เงา การเคลื่อนไหว
- เสียง: เสียงพื้นหลัง จังหวะการพูด
- กลิ่น: กลิ่นสถานที่ กลิ่นตัวละคร
- สัมผัส: อุณหภูมิ พื้นผิว
- รส: ถ้าเกี่ยวข้อง

### 5. จังหวะการเล่า (Pacing)
- ฉากตื่นเต้น: ประโยคสั้น กระชับ
- ฉากอารมณ์: ประโยคยาว ไหลลื่น
- สลับระหว่างบทสนทนา-บรรยาย-ความคิด

### 6. ความขัดแย้งภายใน
ตัวละครต้องมี:
- ความต้องการภายนอก (want) vs ความต้องการภายใน (need)
- ความขัดแย้งภายในใจ

## สไตล์การเขียน
{style_info.get('instructions', '')}"""

        # Add few-shot examples if provided
        if few_shot_examples:
            base_prompt += "\n\n## ตัวอย่างงานเขียนคุณภาพสูง\n"
            base_prompt += "ศึกษาตัวอย่างเหล่านี้เพื่อเข้าใจมาตรฐานที่ต้องการ:\n\n"

            for i, example in enumerate(few_shot_examples, 1):
                # Truncate examples to keep prompt manageable
                input_preview = example['input_context'][:300]
                output_preview = example['output_example'][:600]

                base_prompt += f"### ตัวอย่างที่ {i}\n"
                base_prompt += f"**คำสั่ง:**\n{input_preview}...\n\n"
                base_prompt += f"**ผลงาน:**\n{output_preview}...\n\n"
                base_prompt += f"(คะแนน: {example['quality_score']:.1f}/10)\n\n"

        base_prompt += f"""
## ระบบสรรพนาม
{self._format_pronouns(style_info.get('pronouns', {}))}

## รูปแบบ Output
ตอบเป็น JSON:
```json
{{
    "thai_prose": "<เนื้อหานิยายภาษาไทย อย่างน้อย 6500 ตัวอักษร>",
    "refinement_instructions": {{
        "focus_areas": ["จุดที่ต้องเน้น"],
        "style_adjustments": ["การปรับสไตล์"],
        "specific_fixes": ["สิ่งที่ต้องแก้"],
        "enhancement_suggestions": ["ข้อเสนอแนะเพิ่มเติม"]
    }}
}}
```

## ข้อห้ามเด็ดขาด
- ห้ามเขียนซ้ำประโยคเดิม
- ห้ามบอกอารมณ์ตรงๆ ต้องแสดงผ่านการกระทำ
- ห้ามใช้ภาษาอังกฤษในเนื้อเรื่อง
- ห้ามจบฉากแบบลอยๆ ต้องมี hook หรือ cliffhanger"""

        return base_prompt

    def _format_pronouns(self, pronouns: Dict) -> str:
        """Format pronouns for the prompt"""
        if not pronouns:
            return "Use appropriate Thai pronouns based on character relationships."

        lines = []
        for category, pronoun_set in pronouns.items():
            pronoun_str = ", ".join([f"{k}: {v}" for k, v in pronoun_set.items()])
            lines.append(f"   - {category}: {pronoun_str}")
        return "\n".join(lines)

    def write_scene_with_instructions(
        self,
        scene_instructions: str,
        style: str,
        world_context: str = "",
        previous_feedback: str = "",
        project_id: Optional[int] = None,
        use_few_shot: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Thai prose with refinement instructions for local AI

        Args:
            scene_instructions: Detailed English scene beats from Planner
            style: Writing style
            world_context: Character/world information from RAG
            previous_feedback: Feedback from failed evaluation (for rewrites)
            project_id: Project ID for fetching relevant examples
            use_few_shot: Whether to use few-shot examples (default: True)

        Returns:
            Tuple of (thai_prose, refinement_instructions)
        """
        # Fetch few-shot examples if enabled
        few_shot_examples = []
        if use_few_shot and project_id and os.getenv("FEW_SHOT_ENABLED", "true").lower() == "true":
            try:
                from nodes.database import DatabaseNode
                db = DatabaseNode()
                few_shot_examples = db.get_few_shot_examples(
                    example_type='writer',
                    style=style,
                    query_context=scene_instructions,
                    limit=int(os.getenv("FEW_SHOT_MAX_EXAMPLES_PER_PROMPT", "2")),
                    min_quality=float(os.getenv("FEW_SHOT_MIN_QUALITY", "8.0"))
                )
                if few_shot_examples:
                    print(f"[ClaudeWriter] Using {len(few_shot_examples)} few-shot examples")
            except Exception as e:
                print(f"[ClaudeWriter] Failed to fetch few-shot examples: {e}")

        system_prompt = self._get_system_prompt(style, few_shot_examples)

        input_text = f"""## คำสั่งสำหรับฉากนี้
{scene_instructions}

## ข้อมูลตัวละครและฉาก
{world_context if world_context else "สร้างรายละเอียดที่เหมาะสมตามบริบท"}

## แนวทางการเขียนฉากนี้

### ขั้นตอนที่ 1: วางโครงสร้าง
ก่อนเขียน ให้คิดว่า:
- ฉากนี้เริ่มต้นอย่างไร? (Hook ที่ดึงดูดผู้อ่าน)
- ตัวละครหลักต้องการอะไร?
- อุปสรรคหรือความขัดแย้งคืออะไร?
- ฉากนี้จบอย่างไร? (Cliffhanger หรือ turning point)

### ขั้นตอนที่ 2: เขียนอย่างมีชีวิต
- เริ่มด้วย action หรือ dialogue ไม่ใช่ description
- สลับระหว่าง: บทสนทนา → บรรยาย → ความคิดภายใน
- ใส่รายละเอียดประสาทสัมผัสให้ผู้อ่าน "เห็น" ฉาก
- ให้ตัวละครมี "เสียง" เฉพาะตัว

### ขั้นตอนที่ 3: ตรวจสอบ
- ไม่มีประโยคซ้ำ
- ไม่บอกอารมณ์ตรงๆ (ห้าม "เธอรู้สึก...")
- มี tension หรือ conflict
- จบด้วย hook ที่ทำให้อยากอ่านต่อ
"""

        if previous_feedback:
            input_text += f"""
## สำคัญมาก: ฉบับก่อนถูกปฏิเสธ
Feedback จากผู้ตรวจ:
{previous_feedback}

กรุณาแก้ไขทุกปัญหาที่กล่าวมา และเขียนใหม่ให้ดีกว่าเดิม
"""

        input_text += """
---

เขียนฉากนี้เป็นภาษาไทยที่สวยงาม มีชีวิตชีวา อย่างน้อย 6500 ตัวอักษร
ตอบเป็น JSON format ตามที่กำหนด"""

        result = self.chain.invoke({
            "system_prompt": system_prompt,
            "input": input_text
        })

        return self._parse_response(result)

    def _parse_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Parse Claude's JSON response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                thai_prose = data.get("thai_prose", "")
                instructions = data.get("refinement_instructions", {})
                return self._clean_output(thai_prose), instructions
        except json.JSONDecodeError:
            pass

        # Fallback: return response as prose with empty instructions
        return self._clean_output(response), {
            "focus_areas": ["Review overall quality"],
            "style_adjustments": [],
            "specific_fixes": [],
            "enhancement_suggestions": []
        }

    def _clean_output(self, text: str) -> str:
        """Clean the output to ensure pure Thai prose"""
        lines = text.strip().split('\n')
        cleaned_lines = []

        for line in lines:
            if line.strip().startswith(('Note:', 'Translation:', '---', '##', '*', '```', '{')):
                continue
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()


def create_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for writing (Ollama-based)

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


def create_claude_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for Claude-based writing

    Args:
        state: Current graph state

    Returns:
        Updated state with draft_content, claude_draft, and refinement_instructions
    """
    try:
        writer = ClaudeWriterNode()
    except (ImportError, ValueError) as e:
        # Fallback to Ollama if Claude is not available
        print(f"[Warning] Claude not available: {e}. Using Ollama instead.")
        return create_writer_node(state)

    # Get feedback from previous iteration if exists
    previous_feedback = ""
    if state.get("kpi_report") and state.get("iteration_count", 0) > 0:
        kpi = state["kpi_report"]
        if isinstance(kpi, dict):
            previous_feedback = kpi.get("feedback", "")

    # Generate Thai prose with refinement instructions
    thai_prose, refinement_instructions = writer.write_scene_with_instructions(
        scene_instructions=state.get("scene_instructions", ""),
        style=state.get("style", "modern_thai"),
        world_context=state.get("retrieved_context", ""),
        previous_feedback=previous_feedback
    )

    # Increment iteration count
    new_iteration = state.get("iteration_count", 0) + 1

    return {
        **state,
        "draft_content": thai_prose,
        "claude_draft": thai_prose,
        "refinement_instructions": refinement_instructions,
        "iteration_count": new_iteration
    }
