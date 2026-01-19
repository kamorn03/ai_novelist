"""
AI Novelist Orchestrator - Main Entry Point
Autonomous Thai novel generation using LangGraph
"""
import os
import sys
from typing import Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state import NovelState, create_initial_state, KPIReport, STYLE_GUIDES
from nodes.database import DatabaseNode
from nodes.planner import PlannerNode
from nodes.writer import WriterNode
from nodes.evaluator import EvaluatorNode

load_dotenv()

app = typer.Typer(help="AI Novelist Orchestrator - Generate Thai novels with AI")
console = Console()


# ===========================================
# LangGraph Node Functions
# ===========================================

def initialize_node(state: NovelState) -> NovelState:
    """Initialize project and retrieve context"""
    console.print("[bold blue]Initializing project...[/bold blue]")

    db = DatabaseNode()

    # Create project if not exists
    if state["project_id"] is None:
        project_id = db.create_project(
            title=state["project_title"],
            plot_summary=state["plot_summary"],
            style=state["style"],
            total_chapters=state["total_chapters"]
        )
        state["project_id"] = project_id

    # Retrieve context for current chapter
    context = db.build_context_for_scene(
        project_id=state["project_id"],
        scene_description=state["plot_summary"],
        include_characters=True
    )
    state["retrieved_context"] = context

    console.print(f"[green]Project initialized (ID: {state['project_id']})[/green]")
    return state


def planner_node(state: NovelState) -> NovelState:
    """Plan the scene"""
    console.print(f"\n[bold yellow]Planning Chapter {state['current_chapter']}...[/bold yellow]")

    planner = PlannerNode()

    # Get previous chapters for context
    db = DatabaseNode()
    previous_chapters = []
    if state["project_id"] and state["current_chapter"] > 1:
        previous_chapters = db.get_previous_chapters(
            state["project_id"],
            state["current_chapter"]
        )

    previous_context = ""
    if previous_chapters:
        previous_context = "\n".join([
            f"Chapter {ch['chapter_number']}: {ch['content'][:500]}..."
            for ch in previous_chapters
        ])

    # Generate scene instructions
    instructions = planner.plan_scene(
        plot_summary=state["plot_summary"],
        chapter_number=state["current_chapter"],
        total_chapters=state["total_chapters"],
        style=state["style"],
        previous_context=previous_context,
        world_context=state["retrieved_context"]
    )

    state["scene_instructions"] = instructions
    console.print("[green]Scene planned successfully[/green]")

    return state


def writer_node(state: NovelState) -> NovelState:
    """Write the Thai prose"""
    iteration = state["iteration_count"] + 1
    console.print(f"\n[bold cyan]Writing Thai prose (Iteration {iteration})...[/bold cyan]")

    writer = WriterNode()

    # Get feedback if rewriting
    previous_feedback = ""
    if state["kpi_report"] and iteration > 1:
        kpi = state["kpi_report"]
        if isinstance(kpi, dict):
            previous_feedback = kpi.get("feedback", "")

    # Generate prose
    draft = writer.write_scene(
        scene_instructions=state["scene_instructions"],
        style=state["style"],
        world_context=state["retrieved_context"],
        previous_feedback=previous_feedback
    )

    state["draft_content"] = draft
    state["iteration_count"] = iteration

    console.print(f"[green]Draft generated ({len(draft)} characters)[/green]")

    return state


def evaluator_node(state: NovelState) -> NovelState:
    """Evaluate the draft"""
    console.print("\n[bold magenta]Evaluating draft...[/bold magenta]")

    evaluator = EvaluatorNode()

    # Quick validation
    quick = evaluator.quick_check(state["draft_content"])

    if not quick["is_valid"]:
        kpi = KPIReport(
            consistency_score=0,
            prose_quality_score=0,
            emotional_score=0,
            is_passed=False,
            feedback=f"Invalid draft: Thai content: {quick['has_thai']}, English ratio: {quick['english_ratio']:.1%}"
        )
    else:
        kpi = evaluator.evaluate(
            draft_content=state["draft_content"],
            scene_instructions=state["scene_instructions"],
            world_context=state["retrieved_context"],
            style=state["style"]
        )

    state["kpi_report"] = kpi.model_dump()

    # Display scores
    display_kpi(kpi)

    return state


def save_node(state: NovelState) -> NovelState:
    """Save the finalized chapter"""
    console.print("\n[bold green]Saving chapter...[/bold green]")

    db = DatabaseNode()

    kpi_dict = state["kpi_report"]
    if isinstance(kpi_dict, dict):
        final_score = kpi_dict.get("average_score", 0)
    else:
        final_score = 0

    # Save chapter
    db.save_chapter(
        project_id=state["project_id"],
        chapter_number=state["current_chapter"],
        content=state["draft_content"],
        scene_instructions=state["scene_instructions"],
        title=f"Chapter {state['current_chapter']}",
        final_score=final_score,
        iterations_used=state["iteration_count"]
    )

    # Log KPI
    if kpi_dict:
        db.log_kpi(
            project_id=state["project_id"],
            chapter_number=state["current_chapter"],
            iteration=state["iteration_count"],
            consistency_score=kpi_dict.get("consistency_score", 0),
            prose_quality_score=kpi_dict.get("prose_quality_score", 0),
            emotional_score=kpi_dict.get("emotional_score", 0),
            is_passed=kpi_dict.get("is_passed", False),
            feedback=kpi_dict.get("feedback", ""),
            draft_content=state["draft_content"]
        )

    console.print(f"[green]Chapter {state['current_chapter']} saved successfully![/green]")

    # Move to next chapter or complete
    if state["current_chapter"] < state["total_chapters"]:
        state["current_chapter"] += 1
        state["iteration_count"] = 0
        state["draft_content"] = ""
        state["kpi_report"] = None
        state["should_continue"] = True
    else:
        state["is_complete"] = True
        state["should_continue"] = False
        db.update_project_status(state["project_id"], "completed")

    return state


# ===========================================
# Routing Functions
# ===========================================

def should_rewrite(state: NovelState) -> Literal["writer", "save"]:
    """Determine if draft needs rewriting"""
    kpi = state.get("kpi_report")
    max_iter = state.get("max_iterations", 3)
    current_iter = state.get("iteration_count", 0)

    if kpi is None:
        return "save"

    is_passed = kpi.get("is_passed", False) if isinstance(kpi, dict) else False

    if is_passed:
        console.print("[bold green]Draft PASSED evaluation![/bold green]")
        return "save"
    elif current_iter >= max_iter:
        console.print(f"[bold yellow]Max iterations ({max_iter}) reached. Saving best effort.[/bold yellow]")
        return "save"
    else:
        console.print(f"[bold red]Draft FAILED. Rewriting (attempt {current_iter + 1}/{max_iter})...[/bold red]")
        return "writer"


def should_continue(state: NovelState) -> Literal["planner", "end"]:
    """Determine if should continue to next chapter"""
    if state.get("is_complete", False):
        return "end"
    if state.get("should_continue", True):
        return "planner"
    return "end"


# ===========================================
# Build LangGraph Workflow
# ===========================================

def build_workflow() -> StateGraph:
    """Build the LangGraph workflow"""
    workflow = StateGraph(NovelState)

    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("save", save_node)

    # Set entry point
    workflow.set_entry_point("initialize")

    # Add edges
    workflow.add_edge("initialize", "planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "evaluator")

    # Conditional edges
    workflow.add_conditional_edges(
        "evaluator",
        should_rewrite,
        {
            "writer": "writer",
            "save": "save"
        }
    )

    workflow.add_conditional_edges(
        "save",
        should_continue,
        {
            "planner": "planner",
            "end": END
        }
    )

    return workflow.compile()


# ===========================================
# Helper Functions
# ===========================================

def display_kpi(kpi: KPIReport):
    """Display KPI scores in a table"""
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Status", justify="center")

    def score_status(score: float, threshold: float) -> str:
        return "[green]PASS[/green]" if score >= threshold else "[red]FAIL[/red]"

    min_consistency = float(os.getenv("MIN_CONSISTENCY_SCORE", "7"))
    min_prose = float(os.getenv("MIN_PROSE_QUALITY_SCORE", "7"))
    min_emotional = float(os.getenv("MIN_EMOTIONAL_SCORE", "6"))

    table.add_row(
        "Consistency",
        f"{kpi.consistency_score}/10",
        score_status(kpi.consistency_score, min_consistency)
    )
    table.add_row(
        "Prose Quality",
        f"{kpi.prose_quality_score}/10",
        score_status(kpi.prose_quality_score, min_prose)
    )
    table.add_row(
        "Emotional Impact",
        f"{kpi.emotional_score}/10",
        score_status(kpi.emotional_score, min_emotional)
    )
    table.add_row(
        "Average",
        f"{kpi.average_score}/10",
        "[bold green]PASSED[/bold green]" if kpi.is_passed else "[bold red]FAILED[/bold red]"
    )

    console.print(table)


# ===========================================
# CLI Commands
# ===========================================

@app.command()
def generate(
    title: str = typer.Option(..., "--title", "-t", help="Novel title"),
    plot: str = typer.Option(..., "--plot", "-p", help="Plot summary"),
    style: str = typer.Option(
        "modern_thai",
        "--style", "-s",
        help="Writing style: ancient_chinese, thai_period, modern_thai"
    ),
    chapters: int = typer.Option(1, "--chapters", "-c", help="Number of chapters"),
    max_iterations: int = typer.Option(3, "--max-iter", "-m", help="Max rewrites per chapter")
):
    """Generate a Thai novel"""
    console.print(Panel.fit(
        f"[bold]AI Novelist Orchestrator[/bold]\n\n"
        f"Title: {title}\n"
        f"Style: {STYLE_GUIDES.get(style, {}).get('name', style)}\n"
        f"Chapters: {chapters}",
        title="Starting Novel Generation"
    ))

    # Create initial state
    state = create_initial_state(
        project_title=title,
        plot_summary=plot,
        style=style,
        total_chapters=chapters,
        max_iterations=max_iterations
    )

    # Build and run workflow
    workflow = build_workflow()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating novel...", total=None)

        final_state = workflow.invoke(state)

        progress.update(task, description="[bold green]Complete!")

    # Display final results
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Novel Generation Complete![/bold green]\n\n"
        f"Project ID: {final_state['project_id']}\n"
        f"Chapters Written: {final_state['current_chapter'] if final_state['is_complete'] else final_state['current_chapter'] - 1}",
        title="Results"
    ))


@app.command()
def init_db():
    """Initialize the database with schema"""
    console.print("[bold]Initializing database...[/bold]")

    db = DatabaseNode()
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    db.init_database(schema_path)

    console.print("[bold green]Database initialized successfully![/bold green]")


@app.command()
def add_character(
    project_id: int = typer.Option(..., "--project", "-p", help="Project ID"),
    name: str = typer.Option(..., "--name", "-n", help="Character name"),
    description: str = typer.Option(..., "--desc", "-d", help="Character description"),
    pronouns: str = typer.Option(None, "--pronouns", help="Pronouns JSON (e.g., '{\"self\": \"ข้า\"}')")
):
    """Add a character to the world knowledge"""
    import json

    db = DatabaseNode()

    metadata = {}
    if pronouns:
        metadata["pronouns"] = json.loads(pronouns)

    db.add_world_knowledge(
        project_id=project_id,
        category="character",
        name=name,
        description=description,
        metadata=metadata if metadata else None
    )

    console.print(f"[green]Character '{name}' added to project {project_id}[/green]")


@app.command()
def add_setting(
    project_id: int = typer.Option(..., "--project", "-p", help="Project ID"),
    name: str = typer.Option(..., "--name", "-n", help="Setting name"),
    description: str = typer.Option(..., "--desc", "-d", help="Setting description")
):
    """Add a setting/location to the world knowledge"""
    db = DatabaseNode()

    db.add_world_knowledge(
        project_id=project_id,
        category="setting",
        name=name,
        description=description
    )

    console.print(f"[green]Setting '{name}' added to project {project_id}[/green]")


@app.command()
def show_chapter(
    project_id: int = typer.Option(..., "--project", "-p", help="Project ID"),
    chapter: int = typer.Option(..., "--chapter", "-c", help="Chapter number")
):
    """Display a saved chapter"""
    db = DatabaseNode()
    ch = db.get_chapter(project_id, chapter)

    if ch:
        console.print(Panel(
            ch["content"],
            title=f"Chapter {chapter}: {ch.get('title', 'Untitled')}",
            subtitle=f"Score: {ch.get('final_score', 'N/A')} | Iterations: {ch.get('iterations_used', 'N/A')}"
        ))
    else:
        console.print(f"[red]Chapter {chapter} not found in project {project_id}[/red]")


@app.command()
def list_styles():
    """List available writing styles"""
    table = Table(title="Available Writing Styles")
    table.add_column("Style Key", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")

    for key, info in STYLE_GUIDES.items():
        table.add_row(
            key,
            info["name"],
            info["instructions"][:100].strip() + "..."
        )

    console.print(table)


if __name__ == "__main__":
    app()
