"""
LangGraph State Management for AI Novelist
"""
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field


class KPIReport(BaseModel):
    """Evaluation scores and feedback from Evaluator Node"""
    consistency_score: float = Field(default=0.0, ge=0, le=10, description="Story consistency score (0-10)")
    prose_quality_score: float = Field(default=0.0, ge=0, le=10, description="Thai prose quality score (0-10)")
    emotional_score: float = Field(default=0.0, ge=0, le=10, description="Emotional impact score (0-10)")
    average_score: float = Field(default=0.0, ge=0, le=10, description="Average of all scores")
    is_passed: bool = Field(default=False, description="Whether the draft meets minimum thresholds")
    feedback: str = Field(default="", description="Detailed feedback for improvement")

    def calculate_average(self) -> float:
        """Calculate and set the average score"""
        self.average_score = round(
            (self.consistency_score + self.prose_quality_score + self.emotional_score) / 3, 1
        )
        return self.average_score


class CharacterProfile(BaseModel):
    """Character information for context"""
    name: str
    description: str
    pronouns: dict = Field(default_factory=dict)  # e.g., {"self": "ข้า", "others_call": "ท่าน"}
    traits: List[str] = Field(default_factory=list)
    relationships: dict = Field(default_factory=dict)  # e.g., {"character_b": "พี่ชาย"}


class NovelState(TypedDict):
    """
    Main state for the LangGraph workflow

    This state is passed between nodes and tracks the entire novel generation process.
    """
    # Project Information
    project_id: Optional[int]
    project_title: str
    style: str  # 'ancient_chinese', 'thai_period', 'modern_thai'

    # Plot & Chapter Tracking
    plot_summary: str
    current_chapter: int
    total_chapters: int

    # Scene Planning (Planner Node output)
    scene_instructions: str  # English scene beats

    # Writing (Writer Node output)
    draft_content: str  # Thai prose

    # Evaluation (Evaluator Node output)
    kpi_report: Optional[dict]  # Serialized KPIReport

    # Iteration Control
    iteration_count: int
    max_iterations: int

    # Context from RAG
    retrieved_context: str  # Character/world info for current scene

    # Flow Control
    should_continue: bool  # Whether to continue to next chapter
    is_complete: bool  # Whether the entire novel is complete
    error_message: Optional[str]  # Any error during processing


def create_initial_state(
    project_title: str,
    plot_summary: str,
    style: str = "modern_thai",
    total_chapters: int = 1,
    max_iterations: int = 3
) -> NovelState:
    """
    Create initial state for a new novel project

    Args:
        project_title: Title of the novel
        plot_summary: Overall plot description
        style: Writing style ('ancient_chinese', 'thai_period', 'modern_thai')
        total_chapters: Number of chapters to generate
        max_iterations: Max rewrites per chapter

    Returns:
        Initialized NovelState
    """
    return NovelState(
        project_id=None,
        project_title=project_title,
        style=style,
        plot_summary=plot_summary,
        current_chapter=1,
        total_chapters=total_chapters,
        scene_instructions="",
        draft_content="",
        kpi_report=None,
        iteration_count=0,
        max_iterations=max_iterations,
        retrieved_context="",
        should_continue=True,
        is_complete=False,
        error_message=None
    )


# Style-specific writing guidelines
STYLE_GUIDES = {
    "ancient_chinese": {
        "name": "จีนโบราณ (Ancient Chinese)",
        "pronouns": {
            "male_noble": {"self": "ข้า", "formal": "เปิ่นจวิน", "humble": "ข้าน้อย"},
            "female_noble": {"self": "ข้า", "formal": "เปิ่นกง", "humble": "ข้าน้อย"},
            "servant": {"self": "ข้าน้อย", "master": "ท่าน", "young_master": "เจ้าค่ะ/เจ้าขา"},
            "emperor": {"self": "เจิ้น", "addressing": "ไทเฮา/ปีอ์เซี่ย"}
        },
        "instructions": """
Use formal, poetic Thai with Chinese-style metaphors.
Characters should address each other with proper titles.
Include Chinese-inspired idioms translated elegantly into Thai.
Maintain a sense of grandeur and formality in narration.
"""
    },
    "thai_period": {
        "name": "ไทยยุคเก่า (Thai Period)",
        "pronouns": {
            "royalty": {"self": "หม่อมฉัน", "king": "ใต้ฝ่าละอองธุลีพระบาท"},
            "noble": {"self": "กระผม/ดิฉัน", "superior": "ท่าน", "equal": "คุณ"},
            "commoner": {"self": "ข้าพเจ้า/ฉัน", "superior": "นาย/แม่", "familiar": "พี่/น้อง"}
        },
        "instructions": """
Use elegant Thai prose with period-appropriate vocabulary.
Include traditional Thai expressions and idioms.
Characters should use proper pronouns based on social status.
Descriptions should evoke the atmosphere of historical Thailand.
"""
    },
    "modern_thai": {
        "name": "ไทยสมัยใหม่ (Modern Thai)",
        "pronouns": {
            "general": {"self": "ผม/ฉัน/เรา", "peer": "คุณ/เธอ/นาย", "familiar": "พี่/น้อง/เพื่อน"}
        },
        "instructions": """
Use natural, flowing modern Thai prose.
Dialogue should feel authentic to contemporary speech.
Avoid overly formal or stilted language.
Balance narration with vivid, sensory descriptions.
"""
    }
}
