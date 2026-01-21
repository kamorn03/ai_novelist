"""
AI Novelist Orchestrator - Main Entry Point
Autonomous Thai novel generation using LangGraph
"""
import os
import sys
from typing import Literal

# Fix Windows console encoding for Thai characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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
from nodes.writer import WriterNode, ClaudeWriterNode
from nodes.evaluator import EvaluatorNode
from nodes.refiner import RefinerNode

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
    """Plan the scene using episode blueprint if available"""
    console.print(f"\n[bold yellow]Planning Episode {state['current_chapter']}...[/bold yellow]")

    planner = PlannerNode()
    db = DatabaseNode()

    # Try to get episode blueprint
    blueprint = None
    if state["project_id"]:
        blueprint = db.get_episode_blueprint(state["project_id"], state["current_chapter"])

    if blueprint:
        # Use blueprint-based planning
        console.print(f"[cyan]Using blueprint: {blueprint.get('title', 'Untitled')}[/cyan]")

        # Get episode context (characters, previous episodes, world knowledge)
        episode_context = db.build_episode_context(
            state["project_id"],
            state["current_chapter"]
        )

        # Build previous context summary
        previous_context = ""
        if episode_context['previous_episodes']:
            previous_context = "\n".join([
                f"Episode {ch['chapter_number']}: {ch['content'][:500]}..."
                for ch in episode_context['previous_episodes']
            ])

        # Generate instructions from blueprint
        instructions = planner.plan_from_blueprint(
            blueprint=blueprint,
            characters=episode_context['characters'],
            previous_context=previous_context,
            style=state["style"],
            project_id=state["project_id"]
        )

        # Store additional context
        state["retrieved_context"] = episode_context['world_context']
    else:
        # Fallback to old method (no blueprint)
        console.print("[yellow]No blueprint found, using basic planning...[/yellow]")

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
    """Write the Thai prose using Claude (primary) or Ollama (fallback)"""
    iteration = state["iteration_count"] + 1
    use_claude = state.get("use_claude", True) and os.getenv("USE_CLAUDE_WRITER", "true").lower() == "true"

    if use_claude:
        console.print(f"\n[bold cyan]Writing Thai prose with Claude (Iteration {iteration})...[/bold cyan]")
    else:
        console.print(f"\n[bold cyan]Writing Thai prose with Ollama (Iteration {iteration})...[/bold cyan]")

    # Get feedback if rewriting
    previous_feedback = ""
    if state["kpi_report"] and iteration > 1:
        kpi = state["kpi_report"]
        if isinstance(kpi, dict):
            previous_feedback = kpi.get("feedback", "")

    if use_claude:
        try:
            writer = ClaudeWriterNode()
            # Generate prose with refinement instructions
            draft, refinement_instructions = writer.write_scene_with_instructions(
                scene_instructions=state["scene_instructions"],
                style=state["style"],
                world_context=state["retrieved_context"],
                previous_feedback=previous_feedback,
                project_id=state["project_id"]
            )
            state["claude_draft"] = draft
            state["refinement_instructions"] = refinement_instructions
            console.print(f"[green]Claude draft generated ({len(draft)} characters)[/green]")
            console.print(f"[dim]Refinement instructions: {len(refinement_instructions.get('focus_areas', []))} focus areas[/dim]")
        except Exception as e:
            console.print(f"[yellow]Claude error: {e}. Falling back to Ollama.[/yellow]")
            use_claude = False

    if not use_claude:
        # Fallback to Ollama
        writer = WriterNode()
        draft = writer.write_scene(
            scene_instructions=state["scene_instructions"],
            style=state["style"],
            world_context=state["retrieved_context"],
            previous_feedback=previous_feedback,
            project_id=state["project_id"]
        )
        state["refinement_instructions"] = None
        console.print(f"[green]Ollama draft generated ({len(draft)} characters)[/green]")

    state["draft_content"] = draft
    state["iteration_count"] = iteration

    return state


def refiner_node(state: NovelState) -> NovelState:
    """Refine the draft using local AI based on Claude's instructions"""
    # Skip if no refinement instructions or not using Claude workflow
    if not state.get("refinement_instructions"):
        console.print("[dim]No refinement instructions. Skipping refinement.[/dim]")
        return state

    console.print("\n[bold yellow]Refining prose with local AI...[/bold yellow]")

    refiner = RefinerNode()

    # Use claude_draft as source
    original_draft = state.get("claude_draft") or state.get("draft_content", "")

    if not original_draft:
        console.print("[yellow]No draft to refine.[/yellow]")
        return state

    # Refine the prose
    refined_content = refiner.refine_prose(
        original_draft=original_draft,
        refinement_instructions=state.get("refinement_instructions", {}),
        style=state.get("style", "modern_thai"),
        world_context=state.get("retrieved_context", "")
    )

    state["draft_content"] = refined_content
    console.print(f"[green]Refined draft ({len(refined_content)} characters)[/green]")

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

    # Auto-collect few-shot example if high quality
    auto_collect_threshold = float(os.getenv("FEW_SHOT_AUTO_COLLECT_THRESHOLD", "8.0"))
    if final_score >= auto_collect_threshold:
        try:
            example_id = db.auto_collect_few_shot_example(
                project_id=state["project_id"],
                chapter_number=state["current_chapter"],
                min_quality_threshold=auto_collect_threshold
            )
            if example_id:
                console.print(f"[cyan]✓ Auto-collected few-shot example (ID: {example_id}, score: {final_score:.1f}/10)[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Failed to auto-collect example: {e}[/yellow]")

    # Move to next chapter or complete
    if state["current_chapter"] < state["total_chapters"]:
        state["current_chapter"] += 1
        state["iteration_count"] = 0
        state["draft_content"] = ""
        state["claude_draft"] = ""
        state["refinement_instructions"] = None
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
    """Build the LangGraph workflow with hybrid Claude/Ollama support"""
    workflow = StateGraph(NovelState)

    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("save", save_node)

    # Set entry point
    workflow.set_entry_point("initialize")

    # Add edges
    # Flow: initialize → planner → writer → refiner → evaluator
    workflow.add_edge("initialize", "planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "refiner")
    workflow.add_edge("refiner", "evaluator")

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
    max_iterations: int = typer.Option(3, "--max-iter", "-m", help="Max rewrites per chapter"),
    use_claude: bool = typer.Option(True, "--claude/--no-claude", help="Use Claude API for writing")
):
    """Generate a Thai novel"""
    writer_info = "[cyan]Claude + Local AI[/cyan]" if use_claude else "[yellow]Local AI only[/yellow]"
    console.print(Panel.fit(
        f"[bold]AI Novelist Orchestrator[/bold]\n\n"
        f"Title: {title}\n"
        f"Style: {STYLE_GUIDES.get(style, {}).get('name', style)}\n"
        f"Chapters: {chapters}\n"
        f"Writer: {writer_info}",
        title="Starting Novel Generation"
    ))

    # Create initial state
    state = create_initial_state(
        project_title=title,
        plot_summary=plot,
        style=style,
        total_chapters=chapters,
        max_iterations=max_iterations,
        use_claude=use_claude
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


@app.command()
def import_blueprint(
    blueprint_path: str = typer.Option(..., "--file", "-f", help="Path to blueprint markdown file"),
    project_id: int = typer.Option(None, "--project", "-p", help="Existing project ID (creates new if not provided)"),
    title: str = typer.Option(None, "--title", "-t", help="Project title (required if creating new)")
):
    """Import a blueprint markdown file into a project"""
    from blueprint_importer import parse_blueprint_file, BlueprintImporter

    console.print(Panel.fit(
        f"[bold]Importing Blueprint[/bold]\n\nFile: {blueprint_path}",
        title="Blueprint Import"
    ))

    # Parse blueprint
    console.print("[cyan]Parsing blueprint...[/cyan]")
    try:
        blueprint = parse_blueprint_file(blueprint_path)
        console.print(f"[green]Parsed: {blueprint.title}[/green]")
        console.print(f"  Episodes: {len(blueprint.episodes)}")
        console.print(f"  Characters: {len(blueprint.characters)}")
    except Exception as e:
        console.print(f"[red]Error parsing blueprint: {e}[/red]")
        return

    # Create or use existing project
    db = DatabaseNode()
    if project_id is None:
        # Create new project
        proj_title = title or blueprint.title
        project_id = db.create_project(
            title=proj_title,
            plot_summary=blueprint.concept,
            style="modern_thai",
            total_chapters=len(blueprint.episodes)
        )
        console.print(f"[green]Created project: {proj_title} (ID: {project_id})[/green]")
    else:
        console.print(f"[cyan]Using existing project ID: {project_id}[/cyan]")

    # Import blueprint
    console.print("[cyan]Importing to database...[/cyan]")
    importer = BlueprintImporter()
    result = importer.import_blueprint(blueprint, project_id)

    # Show results
    console.print(f"\n[bold green]Import Complete![/bold green]")
    console.print(f"  Episodes imported: {result['episodes_imported']}")
    console.print(f"  Characters imported: {result['characters_imported']}")
    if result['errors']:
        console.print(f"[yellow]  Errors: {len(result['errors'])}[/yellow]")
        for err in result['errors'][:5]:
            console.print(f"    - {err}")

    console.print(f"\n[bold]Project ID: {project_id}[/bold]")
    console.print("Use this ID to generate episodes.")


@app.command()
def project_status(
    project_id: int = typer.Option(..., "--project", "-p", help="Project ID")
):
    """Show project progress and status"""
    db = DatabaseNode()

    # Get project info
    project = db.get_project(project_id)
    if not project:
        console.print(f"[red]Project {project_id} not found[/red]")
        return

    # Get progress
    progress = db.get_project_progress(project_id)

    # Get all blueprints
    blueprints = db.get_all_episode_blueprints(project_id)

    # Display
    console.print(Panel.fit(
        f"[bold]{project['title']}[/bold]\n"
        f"Status: {project['status']}\n"
        f"Style: {project['style']}",
        title=f"Project {project_id}"
    ))

    console.print(f"\n[bold]Progress:[/bold] {progress['completed_chapters']}/{progress['total_episodes']} "
                  f"({progress['progress_percent']}%)")

    if progress['next_episode']:
        console.print(f"[cyan]Next: Episode {progress['next_episode']['episode_number']} - "
                      f"{progress['next_episode']['title']}[/cyan]")

    # Show episode table
    if blueprints:
        table = Table(title="Episode Blueprints")
        table.add_column("EP", style="cyan", width=4)
        table.add_column("Title", width=30)
        table.add_column("Price", width=10)
        table.add_column("Level", width=6)
        table.add_column("Status", width=10)

        # Get completed chapters
        completed = {ch['chapter_number'] for ch in db.get_previous_chapters(project_id, 999, limit=100)}
        completed.add(progress['total_episodes'])  # Include if last is done

        for bp in blueprints:
            status = "[green]Done[/green]" if bp['episode_number'] in completed else "[dim]Pending[/dim]"
            table.add_row(
                str(bp['episode_number']),
                bp['title'][:28] + ".." if len(bp['title']) > 30 else bp['title'],
                f"{bp['price_coins']} coin" if bp['price_coins'] > 0 else "Free",
                bp['intimacy_level'],
                status
            )

        console.print(table)


@app.command()
def write_episode(
    project_id: int = typer.Option(..., "--project", "-p", help="Project ID"),
    episode: int = typer.Option(None, "--episode", "-e", help="Episode number (auto-selects next if not provided)"),
    max_iterations: int = typer.Option(3, "--max-iter", "-m", help="Max rewrites per episode"),
    use_claude: bool = typer.Option(True, "--claude/--no-claude", help="Use Claude API for writing")
):
    """Write a single episode using its blueprint"""
    db = DatabaseNode()

    # Get project
    project = db.get_project(project_id)
    if not project:
        console.print(f"[red]Project {project_id} not found[/red]")
        return

    # Determine episode to write
    if episode is None:
        progress = db.get_project_progress(project_id)
        if progress['next_episode']:
            episode = progress['next_episode']['episode_number']
        else:
            console.print("[yellow]All episodes completed![/yellow]")
            return

    # Get blueprint
    blueprint = db.get_episode_blueprint(project_id, episode)
    if not blueprint:
        console.print(f"[red]No blueprint found for episode {episode}[/red]")
        return

    writer_info = "[cyan]Claude + Local AI[/cyan]" if use_claude else "[yellow]Local AI only[/yellow]"
    console.print(Panel.fit(
        f"[bold]Writing Episode {episode}[/bold]\n\n"
        f"Title: {blueprint['title']}\n"
        f"Tone: {blueprint['tone']}\n"
        f"Intimacy: {blueprint['intimacy_level']}\n"
        f"Writer: {writer_info}",
        title="Episode Generation"
    ))

    # Create state for single episode
    state = create_initial_state(
        project_title=project['title'],
        plot_summary=project['plot_summary'] or "",
        style=project['style'],
        total_chapters=1,  # Single episode
        max_iterations=max_iterations,
        use_claude=use_claude
    )
    state["project_id"] = project_id
    state["current_chapter"] = episode

    # Build and run workflow
    workflow = build_workflow()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress_bar:
        task = progress_bar.add_task(f"Writing episode {episode}...", total=None)

        final_state = workflow.invoke(state)

        progress_bar.update(task, description="[bold green]Complete!")

    # Show result
    console.print(Panel.fit(
        f"[bold green]Episode {episode} Complete![/bold green]\n\n"
        f"Iterations: {final_state['iteration_count']}\n"
        f"Characters: {len(final_state.get('draft_content', ''))}",
        title="Result"
    ))


@app.command()
def write_all(
    project_id: int = typer.Option(..., "--project", "-p", help="Project ID"),
    start_episode: int = typer.Option(1, "--start", "-s", help="Starting episode number"),
    end_episode: int = typer.Option(None, "--end", "-e", help="Ending episode number (all if not specified)"),
    max_iterations: int = typer.Option(3, "--max-iter", "-m", help="Max rewrites per episode"),
    use_claude: bool = typer.Option(True, "--claude/--no-claude", help="Use Claude API for writing")
):
    """Write multiple episodes sequentially"""
    db = DatabaseNode()

    # Get project
    project = db.get_project(project_id)
    if not project:
        console.print(f"[red]Project {project_id} not found[/red]")
        return

    # Get all blueprints
    blueprints = db.get_all_episode_blueprints(project_id)
    if not blueprints:
        console.print(f"[red]No blueprints found for project {project_id}[/red]")
        return

    # Filter by range
    if end_episode is None:
        end_episode = max(bp['episode_number'] for bp in blueprints)

    episodes_to_write = [
        bp for bp in blueprints
        if start_episode <= bp['episode_number'] <= end_episode
    ]

    writer_info = "[cyan]Claude + Local AI[/cyan]" if use_claude else "[yellow]Local AI only[/yellow]"
    console.print(Panel.fit(
        f"[bold]{project['title']}[/bold]\n\n"
        f"Episodes: {start_episode} to {end_episode}\n"
        f"Total: {len(episodes_to_write)} episodes\n"
        f"Writer: {writer_info}",
        title="Batch Generation"
    ))

    # Build workflow once
    workflow = build_workflow()

    # Write each episode
    for bp in episodes_to_write:
        ep_num = bp['episode_number']
        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]Episode {ep_num}: {bp['title']}[/bold cyan]")
        console.print(f"{'='*60}")

        # Create state
        state = create_initial_state(
            project_title=project['title'],
            plot_summary=project['plot_summary'] or "",
            style=project['style'],
            total_chapters=1,
            max_iterations=max_iterations,
            use_claude=use_claude
        )
        state["project_id"] = project_id
        state["current_chapter"] = ep_num

        try:
            final_state = workflow.invoke(state)
            console.print(f"[green]Episode {ep_num} completed! "
                          f"(Iterations: {final_state['iteration_count']})[/green]")
        except Exception as e:
            console.print(f"[red]Error on episode {ep_num}: {e}[/red]")
            if typer.confirm("Continue with next episode?"):
                continue
            else:
                break

    # Final summary
    progress = db.get_project_progress(project_id)
    console.print(f"\n[bold green]Batch Complete![/bold green]")
    console.print(f"Progress: {progress['completed_chapters']}/{progress['total_episodes']} "
                  f"({progress['progress_percent']}%)")


@app.command()
def export_novel(
    project_id: int = typer.Option(..., "--project", "-p", help="Project ID"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown, txt")
):
    """Export completed chapters to files"""
    import os

    db = DatabaseNode()

    # Get project
    project = db.get_project(project_id)
    if not project:
        console.print(f"[red]Project {project_id} not found[/red]")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all chapters
    blueprints = db.get_all_episode_blueprints(project_id)
    exported = 0

    for bp in blueprints:
        chapter = db.get_chapter(project_id, bp['episode_number'])
        if chapter and chapter.get('content'):
            # Create filename
            safe_title = "".join(c for c in bp['title'] if c.isalnum() or c in " -_").strip()
            filename = f"ep{bp['episode_number']:02d}-{safe_title}.{'md' if format == 'markdown' else 'txt'}"
            filepath = os.path.join(output_dir, filename)

            # Write content
            with open(filepath, 'w', encoding='utf-8') as f:
                if format == 'markdown':
                    f.write(f"# ตอนที่ {bp['episode_number']}: {bp['title']}\n\n")
                    f.write("---\n\n")
                f.write(chapter['content'])
                if format == 'markdown':
                    f.write(f"\n\n---\n\n**จบตอนที่ {bp['episode_number']}**\n")

            exported += 1
            console.print(f"[green]Exported: {filename}[/green]")

    console.print(f"\n[bold green]Exported {exported} chapters to {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
