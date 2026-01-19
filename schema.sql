-- ===========================================
-- AI Novelist Orchestrator - Database Schema
-- PostgreSQL with pgvector extension
-- ===========================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================
-- Table: projects
-- Metadata about novels
-- ===========================================
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    genre VARCHAR(100),
    style VARCHAR(50) DEFAULT 'modern_thai',  -- 'ancient_chinese', 'thai_period', 'modern_thai'
    plot_summary TEXT,
    total_chapters INT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'in_progress',  -- 'in_progress', 'completed', 'paused'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- Table: chapters
-- Finalized chapter content
-- ===========================================
CREATE TABLE IF NOT EXISTS chapters (
    id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    chapter_number INT NOT NULL,
    title VARCHAR(255),
    scene_instructions TEXT,  -- English instructions from Planner
    content TEXT NOT NULL,    -- Final Thai prose
    word_count INT DEFAULT 0,
    final_score DECIMAL(3,1),
    iterations_used INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, chapter_number)
);

-- ===========================================
-- Table: kpi_logs
-- History of evaluation scores
-- ===========================================
CREATE TABLE IF NOT EXISTS kpi_logs (
    id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    chapter_number INT NOT NULL,
    iteration INT NOT NULL,
    consistency_score DECIMAL(3,1),
    prose_quality_score DECIMAL(3,1),
    emotional_score DECIMAL(3,1),
    average_score DECIMAL(3,1),
    is_passed BOOLEAN DEFAULT FALSE,
    feedback TEXT,
    draft_content TEXT,  -- Store draft for reference
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- Table: episode_blueprints
-- Detailed blueprint for each episode
-- ===========================================
CREATE TABLE IF NOT EXISTS episode_blueprints (
    id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    episode_number INT NOT NULL,
    title VARCHAR(255),
    price_tier VARCHAR(50) DEFAULT 'normal',  -- 'free', 'normal', 'premium', 'special'
    price_coins INT DEFAULT 0,
    intimacy_level VARCHAR(10) DEFAULT 'L1',  -- 'L1', 'L2', 'L3', 'L4', 'L5'
    characters_involved TEXT[],               -- Array of character names
    tone VARCHAR(255),                        -- e.g., 'ตลก โรแมนติก'
    selling_points TEXT,                      -- จุดขาย
    cliffhanger TEXT,                         -- Cliffhanger description
    scene_structure TEXT,                     -- Scene 1, Scene 2... structure
    notes TEXT,                               -- Additional notes
    arc_number INT DEFAULT 1,                 -- Which story arc (1-4)
    arc_name VARCHAR(255),                    -- Arc name
    embedding vector(384),                    -- For semantic search
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, episode_number)
);

-- ===========================================
-- Table: world_knowledge (Vector RAG)
-- Character profiles, settings, plot points
-- ===========================================
CREATE TABLE IF NOT EXISTS world_knowledge (
    id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    category VARCHAR(50) NOT NULL,  -- 'character', 'setting', 'plot_point', 'relationship'
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    metadata JSONB,  -- Additional structured data (pronouns, traits, etc.)
    embedding vector(384),  -- Vector for semantic search (MiniLM-L12 produces 384 dims)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- Indexes for performance
-- ===========================================
CREATE INDEX IF NOT EXISTS idx_chapters_project ON chapters(project_id);
CREATE INDEX IF NOT EXISTS idx_kpi_logs_project ON kpi_logs(project_id, chapter_number);
CREATE INDEX IF NOT EXISTS idx_world_knowledge_project ON world_knowledge(project_id);
CREATE INDEX IF NOT EXISTS idx_world_knowledge_category ON world_knowledge(project_id, category);

-- Vector similarity search index (HNSW for faster queries)
CREATE INDEX IF NOT EXISTS idx_world_knowledge_embedding
ON world_knowledge USING hnsw (embedding vector_cosine_ops);

-- Episode blueprints indexes
CREATE INDEX IF NOT EXISTS idx_episode_blueprints_project ON episode_blueprints(project_id);
CREATE INDEX IF NOT EXISTS idx_episode_blueprints_episode ON episode_blueprints(project_id, episode_number);
CREATE INDEX IF NOT EXISTS idx_episode_blueprints_embedding
ON episode_blueprints USING hnsw (embedding vector_cosine_ops);

-- ===========================================
-- Helper function to update timestamps
-- ===========================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for auto-updating timestamps
DROP TRIGGER IF EXISTS trigger_projects_updated ON projects;
CREATE TRIGGER trigger_projects_updated
    BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS trigger_chapters_updated ON chapters;
CREATE TRIGGER trigger_chapters_updated
    BEFORE UPDATE ON chapters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS trigger_world_knowledge_updated ON world_knowledge;
CREATE TRIGGER trigger_world_knowledge_updated
    BEFORE UPDATE ON world_knowledge
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS trigger_episode_blueprints_updated ON episode_blueprints;
CREATE TRIGGER trigger_episode_blueprints_updated
    BEFORE UPDATE ON episode_blueprints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
