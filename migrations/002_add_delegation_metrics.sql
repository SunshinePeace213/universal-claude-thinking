-- Migration 002: Add Delegation Metrics Tables
-- Part of Story 1.2: Request Classification Engine with Delegation Integration
-- Created: 2025-08-07

-- Delegation metrics tracking table
CREATE TABLE IF NOT EXISTS delegation_metrics (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    request_id TEXT NOT NULL,
    classification_type VARCHAR(20),
    confidence_score DECIMAL(3,2),
    delegation_method VARCHAR(20),  -- 'keyword', 'semantic', 'fallback'
    selected_agent VARCHAR(50),
    stage1_latency_ms INTEGER,      -- Keyword matching time
    stage2_latency_ms INTEGER,      -- Semantic matching time  
    stage3_latency_ms INTEGER,      -- PE fallback time
    total_latency_ms INTEGER,
    success BOOLEAN DEFAULT 1,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CHECK (delegation_method IN ('keyword', 'semantic', 'fallback'))
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_delegation_classification 
    ON delegation_metrics(classification_type);
    
CREATE INDEX IF NOT EXISTS idx_delegation_agent 
    ON delegation_metrics(selected_agent);
    
CREATE INDEX IF NOT EXISTS idx_delegation_created 
    ON delegation_metrics(created_at DESC);
    
CREATE INDEX IF NOT EXISTS idx_delegation_method 
    ON delegation_metrics(delegation_method);

-- Classification history for accuracy tracking
CREATE TABLE IF NOT EXISTS classification_history (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    prompt_hash TEXT,
    prompt_text TEXT,
    predicted_type VARCHAR(20),
    actual_type VARCHAR(20),
    confidence DECIMAL(3,2),
    correct BOOLEAN,
    patterns_matched TEXT,  -- JSON array of matched patterns
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CHECK (confidence >= 0 AND confidence <= 1)
);

-- Create index for accuracy calculations
CREATE INDEX IF NOT EXISTS idx_classification_accuracy 
    ON classification_history(correct, created_at);
    
CREATE INDEX IF NOT EXISTS idx_classification_type 
    ON classification_history(predicted_type, actual_type);

-- Agent performance metrics
CREATE TABLE IF NOT EXISTS agent_performance (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    agent_name VARCHAR(50) NOT NULL,
    task_type VARCHAR(20),
    total_delegations INTEGER DEFAULT 0,
    successful_completions INTEGER DEFAULT 0,
    avg_confidence DECIMAL(3,2),
    avg_processing_time_ms INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(agent_name, task_type)
);

-- Confidence factor tracking
CREATE TABLE IF NOT EXISTS confidence_factors (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    request_id TEXT NOT NULL,
    classification_score DECIMAL(3,2),
    keyword_match_score DECIMAL(3,2),
    semantic_similarity_score DECIMAL(3,2),
    context_quality_score DECIMAL(3,2),
    input_clarity_score DECIMAL(3,2),
    overall_confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (request_id) REFERENCES delegation_metrics(request_id)
);

-- Create views for analytics

-- View for delegation method distribution
CREATE VIEW IF NOT EXISTS v_delegation_distribution AS
SELECT 
    delegation_method,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence,
    AVG(total_latency_ms) as avg_latency,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
FROM delegation_metrics
GROUP BY delegation_method;

-- View for classification accuracy by type
CREATE VIEW IF NOT EXISTS v_classification_accuracy AS
SELECT 
    predicted_type,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy_percentage,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time
FROM classification_history
GROUP BY predicted_type;

-- View for agent performance summary
CREATE VIEW IF NOT EXISTS v_agent_performance_summary AS
SELECT 
    agent_name,
    SUM(total_delegations) as total_delegations,
    SUM(successful_completions) as successful_completions,
    AVG(avg_confidence) as overall_avg_confidence,
    AVG(avg_processing_time_ms) as overall_avg_time,
    SUM(successful_completions) * 100.0 / NULLIF(SUM(total_delegations), 0) as success_rate
FROM agent_performance
GROUP BY agent_name
ORDER BY total_delegations DESC;

-- View for hourly metrics
CREATE VIEW IF NOT EXISTS v_hourly_metrics AS
SELECT 
    strftime('%Y-%m-%d %H:00:00', created_at) as hour,
    COUNT(*) as delegations,
    AVG(confidence_score) as avg_confidence,
    AVG(total_latency_ms) as avg_latency,
    SUM(CASE WHEN delegation_method = 'keyword' THEN 1 ELSE 0 END) as keyword_matches,
    SUM(CASE WHEN delegation_method = 'semantic' THEN 1 ELSE 0 END) as semantic_matches,
    SUM(CASE WHEN delegation_method = 'fallback' THEN 1 ELSE 0 END) as fallbacks
FROM delegation_metrics
GROUP BY strftime('%Y-%m-%d %H:00:00', created_at)
ORDER BY hour DESC;

-- Trigger to update agent performance on new delegation
CREATE TRIGGER IF NOT EXISTS update_agent_performance
AFTER INSERT ON delegation_metrics
BEGIN
    INSERT OR REPLACE INTO agent_performance (
        agent_name,
        task_type,
        total_delegations,
        successful_completions,
        avg_confidence,
        avg_processing_time_ms,
        last_updated
    )
    VALUES (
        NEW.selected_agent,
        NEW.classification_type,
        COALESCE((SELECT total_delegations FROM agent_performance 
                 WHERE agent_name = NEW.selected_agent 
                 AND task_type = NEW.classification_type), 0) + 1,
        COALESCE((SELECT successful_completions FROM agent_performance 
                 WHERE agent_name = NEW.selected_agent 
                 AND task_type = NEW.classification_type), 0) + 
                 CASE WHEN NEW.success = 1 THEN 1 ELSE 0 END,
        NEW.confidence_score,
        NEW.total_latency_ms,
        CURRENT_TIMESTAMP
    );
END;

-- Migration metadata
CREATE TABLE IF NOT EXISTS migration_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_name TEXT NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO migration_history (migration_name) 
VALUES ('002_add_delegation_metrics.sql');