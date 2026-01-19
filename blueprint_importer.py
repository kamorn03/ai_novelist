"""
Blueprint Importer - Parse markdown blueprints and import to database
"""
import re
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CharacterInfo:
    """Character information from blueprint"""
    name: str
    role: str  # 'protagonist', 'love_interest', 'supporting', 'antagonist'
    description: str
    age: Optional[str] = None
    occupation: Optional[str] = None
    personality: Optional[str] = None
    relationship_to_protagonist: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeBlueprint:
    """Episode blueprint structure"""
    episode_number: int
    title: str
    price_tier: str = "normal"  # 'free', 'normal', 'premium', 'special'
    price_coins: int = 0
    intimacy_level: str = "L1"  # 'L1' to 'L5'
    characters_involved: List[str] = field(default_factory=list)
    tone: str = ""
    selling_points: str = ""
    cliffhanger: str = ""
    scene_structure: str = ""
    notes: str = ""
    arc_number: int = 1
    arc_name: str = ""


@dataclass
class NovelBlueprint:
    """Complete novel blueprint"""
    title: str
    concept: str = ""
    protagonist: Optional[CharacterInfo] = None
    characters: List[CharacterInfo] = field(default_factory=list)
    episodes: List[EpisodeBlueprint] = field(default_factory=list)
    settings: List[Dict[str, str]] = field(default_factory=list)
    intimacy_levels: Dict[str, str] = field(default_factory=dict)
    pricing_structure: Dict[str, Any] = field(default_factory=dict)


class BlueprintParser:
    """Parse markdown blueprint into structured data"""

    def __init__(self):
        self.current_arc_number = 0
        self.current_arc_name = ""

    def parse(self, markdown_content: str) -> NovelBlueprint:
        """Parse blueprint markdown into NovelBlueprint"""
        blueprint = NovelBlueprint(title="")

        # Extract title
        title_match = re.search(r'^#\s*Blueprint:\s*(.+?)(?:\s*\(\d+\s*ตอน\))?$',
                                markdown_content, re.MULTILINE)
        if title_match:
            blueprint.title = title_match.group(1).strip()

        # Extract concept
        concept_match = re.search(r'##\s*Concept\s*หลัก\s*\n(.*?)(?=\n##|\n---|\Z)',
                                  markdown_content, re.DOTALL)
        if concept_match:
            blueprint.concept = concept_match.group(1).strip()

        # Extract protagonist (นางเอก)
        protagonist = self._extract_protagonist(markdown_content)
        if protagonist:
            blueprint.protagonist = protagonist

        # Extract characters
        blueprint.characters = self._extract_characters(markdown_content)

        # Extract intimacy levels
        blueprint.intimacy_levels = self._extract_intimacy_levels(markdown_content)

        # Extract pricing structure
        blueprint.pricing_structure = self._extract_pricing(markdown_content)

        # Extract episodes
        blueprint.episodes = self._extract_episodes(markdown_content)

        return blueprint

    def _extract_protagonist(self, content: str) -> Optional[CharacterInfo]:
        """Extract protagonist info from นางเอก section"""
        # Look for protagonist table
        protagonist_match = re.search(
            r'###\s*นางเอก\s*\n\|.*?\|.*?\|\s*\n\|[-\s|]+\|\s*\n((?:\|.*?\|.*?\|\s*\n)+)',
            content, re.MULTILINE
        )
        if protagonist_match:
            table_content = protagonist_match.group(1)
            info = self._parse_character_table(table_content)
            if info:
                return CharacterInfo(
                    name=info.get('ชื่อ', ''),
                    role='protagonist',
                    description=info.get('บุคลิก', ''),
                    age=info.get('อายุ', ''),
                    occupation=info.get('คณะ', ''),
                    metadata={'secret': info.get('ความลับ', '')}
                )
        return None

    def _extract_characters(self, content: str) -> List[CharacterInfo]:
        """Extract all characters from ผู้ชายหลัก and ตัวละครรอง sections"""
        characters = []

        # Extract male leads (ผู้ชายหลัก)
        male_section = re.search(
            r'###\s*ผู้ชายหลัก.*?\n\|.*?\|.*?\|.*?\|.*?\|\s*\n\|[-\s|]+\|\s*\n((?:\|.*?\|\s*\n)+)',
            content, re.MULTILINE | re.DOTALL
        )
        if male_section:
            rows = re.findall(r'\|\s*\*\*(.+?)\*\*.*?\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|',
                             male_section.group(1))
            for row in rows:
                name, relationship, personality, selling_point = row
                characters.append(CharacterInfo(
                    name=name.strip(),
                    role='love_interest',
                    description=personality.strip(),
                    relationship_to_protagonist=relationship.strip(),
                    metadata={'selling_point': selling_point.strip()}
                ))

        # Extract supporting characters (ตัวละครรอง)
        supporting_section = re.search(
            r'###\s*ตัวละครรอง\s*\n\|.*?\|.*?\|\s*\n\|[-\s|]+\|\s*\n((?:\|.*?\|\s*\n)+)',
            content, re.MULTILINE
        )
        if supporting_section:
            rows = re.findall(r'\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|',
                             supporting_section.group(1))
            for row in rows:
                name, role_desc = row
                is_antagonist = 'ตัวร้าย' in role_desc or 'antagonist' in role_desc.lower()
                characters.append(CharacterInfo(
                    name=name.strip(),
                    role='antagonist' if is_antagonist else 'supporting',
                    description=role_desc.strip()
                ))

        return characters

    def _extract_intimacy_levels(self, content: str) -> Dict[str, str]:
        """Extract intimacy level definitions"""
        levels = {}
        level_section = re.search(
            r'##\s*สรุประดับความใกล้ชิด.*?\n\|.*?\|.*?\|.*?\|\s*\n\|[-\s|]+\|\s*\n((?:\|.*?\|\s*\n)+)',
            content, re.MULTILINE
        )
        if level_section:
            rows = re.findall(r'\|\s*(L\d)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|',
                             level_section.group(1))
            for level, meaning, usage in rows:
                levels[level] = f"{meaning.strip()} - {usage.strip()}"
        return levels

    def _extract_pricing(self, content: str) -> Dict[str, Any]:
        """Extract pricing structure"""
        pricing = {}
        pricing_section = re.search(
            r'##\s*โครงสร้างการขาย.*?\n\|.*?\|.*?\|.*?\|.*?\|\s*\n\|[-\s|]+\|\s*\n((?:\|.*?\|\s*\n)+)',
            content, re.MULTILINE
        )
        if pricing_section:
            rows = re.findall(r'\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|\s*(\d+).*?\|\s*(.+?)\s*\|',
                             pricing_section.group(1))
            for tier, episodes, price, reason in rows:
                pricing[tier.strip()] = {
                    'episodes': episodes.strip(),
                    'price': int(price),
                    'reason': reason.strip()
                }
        return pricing

    def _extract_episodes(self, content: str) -> List[EpisodeBlueprint]:
        """Extract all episode blueprints"""
        episodes = []

        # Find arc headers and episode sections
        arc_pattern = r'###\s*Arc\s*(\d+):\s*(.+?)\s*\(ตอน\s*(\d+)-(\d+)\)'
        episode_pattern = r'####\s*ตอนที่\s*(\d+):\s*(.+?)(?:\s*⭐+)?\s*(?:\(.*?\))?\s*\n'

        # Split content by Arc headers
        parts = re.split(r'(###\s*Arc\s*\d+:)', content)

        current_arc = 1
        current_arc_name = ""

        for i, part in enumerate(parts):
            # Check if this is an arc header
            arc_match = re.match(r'###\s*Arc\s*(\d+):', part)
            if arc_match:
                current_arc = int(arc_match.group(1))
                # Get arc name from next part
                if i + 1 < len(parts):
                    arc_name_match = re.match(r'\s*(.+?)\s*\(ตอน', parts[i + 1])
                    if arc_name_match:
                        current_arc_name = arc_name_match.group(1).strip()
                continue

            # Find episodes in this part
            ep_matches = list(re.finditer(episode_pattern, part))

            for j, ep_match in enumerate(ep_matches):
                ep_num = int(ep_match.group(1))
                ep_title = ep_match.group(2).strip()

                # Get episode content (until next episode or end)
                start_pos = ep_match.end()
                if j + 1 < len(ep_matches):
                    end_pos = ep_matches[j + 1].start()
                else:
                    end_pos = len(part)

                ep_content = part[start_pos:end_pos]

                # Parse episode details
                episode = self._parse_episode_content(ep_num, ep_title, ep_content)
                episode.arc_number = current_arc
                episode.arc_name = current_arc_name

                episodes.append(episode)

        return sorted(episodes, key=lambda x: x.episode_number)

    def _parse_episode_content(self, ep_num: int, title: str, content: str) -> EpisodeBlueprint:
        """Parse individual episode content"""
        episode = EpisodeBlueprint(
            episode_number=ep_num,
            title=title
        )

        # Parse table
        table_match = re.search(
            r'\|.*?หัวข้อ.*?\|.*?รายละเอียด.*?\|\s*\n\|[-\s|]+\|\s*\n((?:\|.*?\|\s*\n)+)',
            content
        )
        if table_match:
            table_content = table_match.group(1)

            # Extract fields
            fields = {
                'ราคา': r'\|\s*\*\*ราคา\*\*\s*\|\s*(.+?)\s*\|',
                'ระดับใกล้ชิด': r'\|\s*\*\*ระดับใกล้ชิด\*\*\s*\|\s*(.+?)\s*\|',
                'คู่ที่เจอ': r'\|\s*\*\*คู่ที่เจอ\*\*\s*\|\s*(.+?)\s*\|',
                'โทน': r'\|\s*\*\*โทน\*\*\s*\|\s*(.+?)\s*\|',
                'จุดขาย': r'\|\s*\*\*จุดขาย\*\*\s*\|\s*(.+?)\s*\|',
                'Cliffhanger': r'\|\s*\*\*Cliffhanger\*\*\s*\|\s*(.+?)\s*\|',
            }

            for field_name, pattern in fields.items():
                match = re.search(pattern, table_content)
                if match:
                    value = match.group(1).strip()
                    if field_name == 'ราคา':
                        episode.price_tier, episode.price_coins = self._parse_price(value)
                    elif field_name == 'ระดับใกล้ชิด':
                        episode.intimacy_level = self._parse_intimacy(value)
                    elif field_name == 'คู่ที่เจอ':
                        episode.characters_involved = [c.strip() for c in value.split(',')]
                    elif field_name == 'โทน':
                        episode.tone = value
                    elif field_name == 'จุดขาย':
                        episode.selling_points = value
                    elif field_name == 'Cliffhanger':
                        episode.cliffhanger = value

        # Extract scene structure
        structure_match = re.search(r'\*\*โครงสร้าง:\*\*\s*\n```\s*(.*?)\s*```',
                                    content, re.DOTALL)
        if structure_match:
            episode.scene_structure = structure_match.group(1).strip()

        # Extract notes
        notes_match = re.search(r'\*\*หมายเหตุ:\*\*\s*(.+?)(?=\n---|\n####|\Z)',
                                content, re.DOTALL)
        if notes_match:
            episode.notes = notes_match.group(1).strip()

        return episode

    def _parse_price(self, value: str) -> tuple:
        """Parse price string into tier and coins"""
        value_lower = value.lower()
        if 'ฟรี' in value_lower or 'free' in value_lower or value == '0':
            return 'free', 0

        coin_match = re.search(r'(\d+)\s*coin', value_lower)
        if coin_match:
            coins = int(coin_match.group(1))
            if coins >= 10:
                return 'special', coins
            elif coins >= 8:
                return 'premium', coins
            else:
                return 'normal', coins

        return 'normal', 5

    def _parse_intimacy(self, value: str) -> str:
        """Parse intimacy level"""
        match = re.search(r'L(\d)', value)
        if match:
            level = int(match.group(1))
            # If range like "L3-L4", take higher
            range_match = re.search(r'L(\d)-L(\d)', value)
            if range_match:
                level = int(range_match.group(2))
            return f"L{level}"
        return "L1"

    def _parse_character_table(self, table_content: str) -> Dict[str, str]:
        """Parse a markdown table into dict"""
        result = {}
        rows = re.findall(r'\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|', table_content)
        for key, value in rows:
            result[key.strip()] = value.strip()
        return result


class BlueprintImporter:
    """Import parsed blueprint into database"""

    def __init__(self):
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._embedding_model = None
        self.db_config = {
            "host": os.getenv("DATABASE_HOST", "localhost"),
            "port": os.getenv("DATABASE_PORT", "5432"),
            "database": os.getenv("DATABASE_NAME", "ai_novelist"),
            "user": os.getenv("DATABASE_USER", "postgres"),
            "password": os.getenv("DATABASE_PASSWORD", "")
        }

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def import_blueprint(self, blueprint: NovelBlueprint, project_id: int) -> Dict[str, Any]:
        """
        Import a complete blueprint into database

        Args:
            blueprint: Parsed NovelBlueprint
            project_id: Existing project ID to link to

        Returns:
            Summary of imported items
        """
        import psycopg2
        from psycopg2.extras import execute_values

        summary = {
            'episodes_imported': 0,
            'characters_imported': 0,
            'errors': []
        }

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Import episodes
            for episode in blueprint.episodes:
                try:
                    # Generate embedding for episode
                    embed_text = f"{episode.title} {episode.tone} {episode.selling_points} {episode.scene_structure}"
                    embedding = self.embedding_model.encode(embed_text).tolist()

                    cur.execute("""
                        INSERT INTO episode_blueprints
                        (project_id, episode_number, title, price_tier, price_coins,
                         intimacy_level, characters_involved, tone, selling_points,
                         cliffhanger, scene_structure, notes, arc_number, arc_name, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (project_id, episode_number)
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            price_tier = EXCLUDED.price_tier,
                            price_coins = EXCLUDED.price_coins,
                            intimacy_level = EXCLUDED.intimacy_level,
                            characters_involved = EXCLUDED.characters_involved,
                            tone = EXCLUDED.tone,
                            selling_points = EXCLUDED.selling_points,
                            cliffhanger = EXCLUDED.cliffhanger,
                            scene_structure = EXCLUDED.scene_structure,
                            notes = EXCLUDED.notes,
                            arc_number = EXCLUDED.arc_number,
                            arc_name = EXCLUDED.arc_name,
                            embedding = EXCLUDED.embedding
                    """, (
                        project_id, episode.episode_number, episode.title,
                        episode.price_tier, episode.price_coins, episode.intimacy_level,
                        episode.characters_involved, episode.tone, episode.selling_points,
                        episode.cliffhanger, episode.scene_structure, episode.notes,
                        episode.arc_number, episode.arc_name, embedding
                    ))
                    summary['episodes_imported'] += 1
                except Exception as e:
                    summary['errors'].append(f"Episode {episode.episode_number}: {str(e)}")

            # Import protagonist
            if blueprint.protagonist:
                try:
                    self._import_character(cur, project_id, blueprint.protagonist)
                    summary['characters_imported'] += 1
                except Exception as e:
                    summary['errors'].append(f"Protagonist: {str(e)}")

            # Import other characters
            for char in blueprint.characters:
                try:
                    self._import_character(cur, project_id, char)
                    summary['characters_imported'] += 1
                except Exception as e:
                    summary['errors'].append(f"Character {char.name}: {str(e)}")

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            summary['errors'].append(f"Database error: {str(e)}")

        return summary

    def _import_character(self, cursor, project_id: int, char: CharacterInfo):
        """Import a single character to world_knowledge"""
        embed_text = f"{char.name}: {char.description}"
        embedding = self.embedding_model.encode(embed_text).tolist()

        metadata = {
            'role': char.role,
            'age': char.age,
            'occupation': char.occupation,
            'personality': char.personality,
            'relationship_to_protagonist': char.relationship_to_protagonist,
            **char.metadata
        }
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        cursor.execute("""
            INSERT INTO world_knowledge
            (project_id, category, name, description, metadata, embedding)
            VALUES (%s, 'character', %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            project_id, char.name, char.description,
            json.dumps(metadata, ensure_ascii=False), embedding
        ))


def parse_blueprint_file(filepath: str) -> NovelBlueprint:
    """Parse a blueprint markdown file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    parser = BlueprintParser()
    return parser.parse(content)


def import_blueprint_to_project(blueprint_path: str, project_id: int) -> Dict[str, Any]:
    """
    Main function to import a blueprint file to a project

    Args:
        blueprint_path: Path to blueprint markdown file
        project_id: Project ID in database

    Returns:
        Import summary
    """
    # Parse blueprint
    blueprint = parse_blueprint_file(blueprint_path)

    # Import to database
    importer = BlueprintImporter()
    return importer.import_blueprint(blueprint, project_id)


if __name__ == "__main__":
    # Test parsing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python blueprint_importer.py <blueprint.md> [project_id]")
        sys.exit(1)

    blueprint_path = sys.argv[1]

    # Parse and print summary
    blueprint = parse_blueprint_file(blueprint_path)

    print(f"\n{'='*60}")
    print(f"Blueprint: {blueprint.title}")
    print(f"{'='*60}")
    print(f"Concept: {blueprint.concept[:100]}..." if blueprint.concept else "No concept")
    print(f"\nProtagonist: {blueprint.protagonist.name if blueprint.protagonist else 'Not found'}")
    print(f"Characters: {len(blueprint.characters)}")
    print(f"Episodes: {len(blueprint.episodes)}")

    if blueprint.episodes:
        print(f"\n{'='*60}")
        print("Episodes Summary:")
        print(f"{'='*60}")
        for ep in blueprint.episodes:
            print(f"  {ep.episode_number:2d}. {ep.title[:30]:30s} | {ep.price_tier:8s} | {ep.intimacy_level}")

    # Import if project_id provided
    if len(sys.argv) >= 3:
        project_id = int(sys.argv[2])
        print(f"\n{'='*60}")
        print(f"Importing to project {project_id}...")
        print(f"{'='*60}")

        result = import_blueprint_to_project(blueprint_path, project_id)
        print(f"Episodes imported: {result['episodes_imported']}")
        print(f"Characters imported: {result['characters_imported']}")
        if result['errors']:
            print(f"Errors: {result['errors']}")
