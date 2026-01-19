"""
Database Node - PostgreSQL + pgvector operations
Handles project management, chapter storage, and Vector RAG
"""
import os
import json
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class DatabaseNode:
    """
    Database operations for AI Novelist

    Handles:
    - Project CRUD
    - Chapter storage
    - KPI logging
    - Vector RAG for world knowledge
    """

    def __init__(self):
        self.db_config = {
            "host": os.getenv("DATABASE_HOST", "localhost"),
            "port": os.getenv("DATABASE_PORT", "5432"),
            "database": os.getenv("DATABASE_NAME", "ai_novelist"),
            "user": os.getenv("DATABASE_USER", "postgres"),
            "password": os.getenv("DATABASE_PASSWORD", "")
        }
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._embedding_model = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(**self.db_config)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self, schema_path: str = "schema.sql"):
        """Initialize database with schema"""
        # Read schema file
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
        print("[DB] Database initialized successfully")

    # ===========================================
    # Project Operations
    # ===========================================

    def create_project(
        self,
        title: str,
        plot_summary: str,
        genre: str = "fiction",
        style: str = "modern_thai",
        total_chapters: int = 1
    ) -> int:
        """Create a new novel project"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO projects (title, genre, style, plot_summary, total_chapters)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (title, genre, style, plot_summary, total_chapters))
                project_id = cur.fetchone()[0]
        print(f"[DB] Created project: {title} (ID: {project_id})")
        return project_id

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get project by ID"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
                return dict(cur.fetchone()) if cur.rowcount > 0 else None

    def update_project_status(self, project_id: int, status: str):
        """Update project status"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE projects SET status = %s WHERE id = %s",
                    (status, project_id)
                )

    # ===========================================
    # Chapter Operations
    # ===========================================

    def save_chapter(
        self,
        project_id: int,
        chapter_number: int,
        content: str,
        scene_instructions: str = "",
        title: str = "",
        final_score: float = 0.0,
        iterations_used: int = 1
    ) -> int:
        """Save finalized chapter content"""
        word_count = len(content)

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chapters
                    (project_id, chapter_number, title, scene_instructions, content, word_count, final_score, iterations_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_id, chapter_number)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        scene_instructions = EXCLUDED.scene_instructions,
                        content = EXCLUDED.content,
                        word_count = EXCLUDED.word_count,
                        final_score = EXCLUDED.final_score,
                        iterations_used = EXCLUDED.iterations_used
                    RETURNING id
                """, (project_id, chapter_number, title, scene_instructions, content, word_count, final_score, iterations_used))
                chapter_id = cur.fetchone()[0]
        print(f"[DB] Saved chapter {chapter_number} (ID: {chapter_id})")
        return chapter_id

    def get_chapter(self, project_id: int, chapter_number: int) -> Optional[Dict[str, Any]]:
        """Get specific chapter"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM chapters
                    WHERE project_id = %s AND chapter_number = %s
                """, (project_id, chapter_number))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_previous_chapters(self, project_id: int, current_chapter: int, limit: int = 3) -> List[Dict[str, Any]]:
        """Get previous chapters for context"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT chapter_number, title, content
                    FROM chapters
                    WHERE project_id = %s AND chapter_number < %s
                    ORDER BY chapter_number DESC
                    LIMIT %s
                """, (project_id, current_chapter, limit))
                return [dict(row) for row in cur.fetchall()]

    # ===========================================
    # KPI Logging
    # ===========================================

    def log_kpi(
        self,
        project_id: int,
        chapter_number: int,
        iteration: int,
        consistency_score: float,
        prose_quality_score: float,
        emotional_score: float,
        is_passed: bool,
        feedback: str,
        draft_content: str
    ) -> int:
        """Log evaluation scores"""
        average_score = round((consistency_score + prose_quality_score + emotional_score) / 3, 1)

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO kpi_logs
                    (project_id, chapter_number, iteration, consistency_score, prose_quality_score, emotional_score, average_score, is_passed, feedback, draft_content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (project_id, chapter_number, iteration, consistency_score, prose_quality_score, emotional_score, average_score, is_passed, feedback, draft_content))
                log_id = cur.fetchone()[0]
        return log_id

    # ===========================================
    # World Knowledge (Vector RAG)
    # ===========================================

    def add_world_knowledge(
        self,
        project_id: int,
        category: str,
        name: str,
        description: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add world knowledge with vector embedding"""
        # Generate embedding
        embedding_text = f"{name}: {description}"
        embedding = self.embedding_model.encode(embedding_text).tolist()

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO world_knowledge
                    (project_id, category, name, description, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    project_id, category, name, description,
                    json.dumps(metadata) if metadata else None,
                    embedding
                ))
                knowledge_id = cur.fetchone()[0]
        print(f"[DB] Added world knowledge: {name} ({category})")
        return knowledge_id

    def search_world_knowledge(
        self,
        project_id: int,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search world knowledge using vector similarity"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if category:
                    cur.execute("""
                        SELECT id, category, name, description, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM world_knowledge
                        WHERE project_id = %s AND category = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, project_id, category, query_embedding, limit))
                else:
                    cur.execute("""
                        SELECT id, category, name, description, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM world_knowledge
                        WHERE project_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, project_id, query_embedding, limit))

                results = []
                for row in cur.fetchall():
                    item = dict(row)
                    if item.get("metadata"):
                        item["metadata"] = json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]
                    results.append(item)
                return results

    def get_characters(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all characters for a project"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, description, metadata
                    FROM world_knowledge
                    WHERE project_id = %s AND category = 'character'
                    ORDER BY name
                """, (project_id,))
                results = []
                for row in cur.fetchall():
                    item = dict(row)
                    if item.get("metadata"):
                        item["metadata"] = json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]
                    results.append(item)
                return results

    def build_context_for_scene(
        self,
        project_id: int,
        scene_description: str,
        include_characters: bool = True
    ) -> str:
        """
        Build context string for Writer node

        Retrieves relevant world knowledge and formats it for the prompt.
        """
        context_parts = []

        # Search for relevant world knowledge
        relevant_knowledge = self.search_world_knowledge(
            project_id, scene_description, limit=5
        )

        if relevant_knowledge:
            context_parts.append("=== Relevant World Knowledge ===")
            for item in relevant_knowledge:
                context_parts.append(f"\n[{item['category'].upper()}] {item['name']}")
                context_parts.append(f"Description: {item['description']}")
                if item.get("metadata"):
                    if "pronouns" in item["metadata"]:
                        context_parts.append(f"Pronouns: {item['metadata']['pronouns']}")
                    if "relationships" in item["metadata"]:
                        context_parts.append(f"Relationships: {item['metadata']['relationships']}")

        # Get all characters if requested
        if include_characters:
            characters = self.get_characters(project_id)
            if characters:
                context_parts.append("\n=== Character Reference ===")
                for char in characters:
                    context_parts.append(f"\n- {char['name']}: {char['description'][:100]}...")
                    if char.get("metadata", {}).get("pronouns"):
                        context_parts.append(f"  Pronouns: {char['metadata']['pronouns']}")

        return "\n".join(context_parts) if context_parts else "No specific context available."

    # ===========================================
    # Episode Blueprint Operations
    # ===========================================

    def get_episode_blueprint(self, project_id: int, episode_number: int) -> Optional[Dict[str, Any]]:
        """Get blueprint for a specific episode"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM episode_blueprints
                    WHERE project_id = %s AND episode_number = %s
                """, (project_id, episode_number))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_all_episode_blueprints(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all episode blueprints for a project"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM episode_blueprints
                    WHERE project_id = %s
                    ORDER BY episode_number
                """, (project_id,))
                return [dict(row) for row in cur.fetchall()]

    def get_episodes_by_arc(self, project_id: int, arc_number: int) -> List[Dict[str, Any]]:
        """Get all episodes in a specific arc"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM episode_blueprints
                    WHERE project_id = %s AND arc_number = %s
                    ORDER BY episode_number
                """, (project_id, arc_number))
                return [dict(row) for row in cur.fetchall()]

    def search_episode_blueprints(
        self,
        project_id: int,
        query: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Search episode blueprints using vector similarity"""
        query_embedding = self.embedding_model.encode(query).tolist()

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, episode_number, title, tone, selling_points,
                           scene_structure, cliffhanger,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM episode_blueprints
                    WHERE project_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, project_id, query_embedding, limit))
                return [dict(row) for row in cur.fetchall()]

    def build_episode_context(
        self,
        project_id: int,
        episode_number: int
    ) -> Dict[str, Any]:
        """
        Build complete context for writing an episode

        Returns dict with:
        - blueprint: Episode blueprint details
        - characters: Characters involved in this episode
        - previous_episodes: Summary of previous episodes
        - world_context: Relevant world knowledge
        """
        context = {
            'blueprint': None,
            'characters': [],
            'previous_episodes': [],
            'world_context': ""
        }

        # Get episode blueprint
        blueprint = self.get_episode_blueprint(project_id, episode_number)
        if blueprint:
            context['blueprint'] = blueprint

            # Get characters involved
            if blueprint.get('characters_involved'):
                for char_name in blueprint['characters_involved']:
                    char_info = self.search_world_knowledge(
                        project_id, char_name, category='character', limit=1
                    )
                    if char_info:
                        context['characters'].extend(char_info)

            # Build world context from scene structure
            if blueprint.get('scene_structure'):
                context['world_context'] = self.build_context_for_scene(
                    project_id,
                    blueprint['scene_structure'],
                    include_characters=True
                )

        # Get previous episodes
        if episode_number > 1:
            context['previous_episodes'] = self.get_previous_chapters(
                project_id, episode_number, limit=2
            )

        return context

    def get_project_progress(self, project_id: int) -> Dict[str, Any]:
        """Get project writing progress"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get total blueprints
                cur.execute("""
                    SELECT COUNT(*) as total_episodes FROM episode_blueprints
                    WHERE project_id = %s
                """, (project_id,))
                total = cur.fetchone()['total_episodes']

                # Get completed chapters
                cur.execute("""
                    SELECT COUNT(*) as completed_chapters FROM chapters
                    WHERE project_id = %s
                """, (project_id,))
                completed = cur.fetchone()['completed_chapters']

                # Get next episode to write
                cur.execute("""
                    SELECT eb.episode_number, eb.title
                    FROM episode_blueprints eb
                    LEFT JOIN chapters c ON eb.project_id = c.project_id
                        AND eb.episode_number = c.chapter_number
                    WHERE eb.project_id = %s AND c.id IS NULL
                    ORDER BY eb.episode_number
                    LIMIT 1
                """, (project_id,))
                next_ep = cur.fetchone()

                return {
                    'total_episodes': total,
                    'completed_chapters': completed,
                    'progress_percent': round((completed / total * 100) if total > 0 else 0, 1),
                    'next_episode': dict(next_ep) if next_ep else None
                }
