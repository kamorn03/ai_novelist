-- Migration 001: Add Few-Shot Learning Infrastructure
-- Description: Add tables for storing high-quality examples for few-shot learning
-- Created: 2026-01-20

-- ===========================================
-- Table: few_shot_examples
-- Purpose: Store high-quality input/output pairs for few-shot prompting
-- ===========================================

CREATE TABLE IF NOT EXISTS few_shot_examples (
    id SERIAL PRIMARY KEY,

    -- Project association (optional - examples can be project-specific or global)
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,

    -- Categorization
    example_type VARCHAR(50) NOT NULL,       -- 'writer', 'planner', 'refiner'
    style VARCHAR(50) NOT NULL,              -- 'ancient_chinese', 'thai_period', 'modern_thai'

    -- Input/Output pair
    input_context TEXT NOT NULL,             -- Scene instructions, plot summary, or context
    output_example TEXT NOT NULL,            -- High-quality Thai prose or instructions

    -- Source tracking
    source_chapter_id INT REFERENCES chapters(id) ON DELETE SET NULL,

    -- Quality metrics
    quality_score DECIMAL(3,1),              -- Average KPI score (0-10)
    kpi_consistency DECIMAL(3,1),            -- Consistency score
    kpi_prose DECIMAL(3,1),                  -- Prose quality score
    kpi_emotional DECIMAL(3,1),              -- Emotional impact score

    -- Usage tracking
    usage_count INT DEFAULT 0,               -- How many times used in prompts
    last_used_at TIMESTAMP,                  -- When last used
    is_active BOOLEAN DEFAULT TRUE,          -- Can be deactivated if quality degrades

    -- Vector search for semantic retrieval
    embedding vector(384),                   -- MiniLM-L12 embeddings (384 dimensions)

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- Indexes for Performance
-- ===========================================

-- Composite index for filtering by type, style, and active status
CREATE INDEX IF NOT EXISTS idx_few_shot_type_style
ON few_shot_examples(example_type, style, is_active);

-- Index for quality-based retrieval (high-quality examples first)
CREATE INDEX IF NOT EXISTS idx_few_shot_quality
ON few_shot_examples(quality_score DESC)
WHERE is_active = TRUE;

-- HNSW index for fast vector similarity search
CREATE INDEX IF NOT EXISTS idx_few_shot_embedding
ON few_shot_examples
USING hnsw (embedding vector_cosine_ops);

-- Index for source tracking
CREATE INDEX IF NOT EXISTS idx_few_shot_source
ON few_shot_examples(source_chapter_id)
WHERE source_chapter_id IS NOT NULL;

-- Index for project-specific retrieval
CREATE INDEX IF NOT EXISTS idx_few_shot_project
ON few_shot_examples(project_id)
WHERE project_id IS NOT NULL;

-- ===========================================
-- Trigger: Update updated_at on modification
-- ===========================================

-- Reuse existing update_updated_at function if it exists, or create it
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc
        WHERE proname = 'update_updated_at'
    ) THEN
        CREATE FUNCTION update_updated_at()
        RETURNS TRIGGER AS $func$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $func$ LANGUAGE plpgsql;
    END IF;
END $$;

-- Create trigger for few_shot_examples
DROP TRIGGER IF EXISTS trigger_few_shot_examples_updated ON few_shot_examples;
CREATE TRIGGER trigger_few_shot_examples_updated
    BEFORE UPDATE ON few_shot_examples
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ===========================================
-- Comments for Documentation
-- ===========================================

COMMENT ON TABLE few_shot_examples IS 'High-quality examples for few-shot learning in prompts';
COMMENT ON COLUMN few_shot_examples.example_type IS 'Type of example: writer, planner, or refiner';
COMMENT ON COLUMN few_shot_examples.input_context IS 'Input prompt or context that led to high-quality output';
COMMENT ON COLUMN few_shot_examples.output_example IS 'The high-quality output to use as example';
COMMENT ON COLUMN few_shot_examples.quality_score IS 'Average quality score from evaluator (0-10)';
COMMENT ON COLUMN few_shot_examples.embedding IS 'Vector embedding for semantic similarity search';
COMMENT ON COLUMN few_shot_examples.usage_count IS 'Number of times this example has been used in prompts';

-- ===========================================
-- Validation Check
-- ===========================================

-- Verify table was created successfully
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'few_shot_examples'
    ) THEN
        RAISE NOTICE 'Migration 001 completed successfully: few_shot_examples table created';
    ELSE
        RAISE EXCEPTION 'Migration 001 failed: few_shot_examples table not found';
    END IF;
END $$;
