BEGIN;
ROLLBACK;
COMMIT;

-- Table: token_usage
CREATE TABLE token_usage (
	
	id SERIAL PRIMARY KEY,
	
	session_id VARCHAR(255) NOT NULL,
	user_id VARCHAR(255),
	
	created_at TIMESTAMP DEFAULT NOW(),
	processing_time_ms FLOAT,
	
	total_tokens INTEGER NOT NULL,
	supervisor_input_tokens INTEGER,
	supervisor_output_tokens INTEGER,
	supervisor_cache_read_tokens INTEGER DEFAULT 0,
	supervisor_reasoning_tokens INTEGER DEFAULT 0,
	
	user_question TEXT
)

-- Detail table: token_usage_agent_detail
CREATE TABLE token_usage_agent_detail (
	id SERIAL PRIMARY KEY,
	usage_id INTEGER NOT NULL REFERENCES token_usage(id) ON DELETE CASCADE,
	
	agent_name VARCHAR(50) NOT NULL,
	token_type VARCHAR(20) NOT NULL,
	
	input_tokens INTEGER NOT NULL,
	output_tokens INTEGER NOT NULL,
	total_tokens INTEGER NOT NULL,
	cache_read_tokens INTEGER DEFAULT 0,
	reasoning_tokens INTEGER DEFAULT 0
)

CREATE INDEX idx_token_usage_user_id ON token_usage(user_id);
CREATE INDEX idx_token_usage_session_id ON token_usage(session_id);
CREATE INDEX idx_token_usage_created_at ON token_usage(created_at DESC);
CREATE INDEX idx_agent_detail_usage_id ON token_usage_agent_detail(usage_id);
CREATE INDEX idx_agent_detail_agent_name ON token_usage_agent_detail(agent_name);
CREATE INDEX idx_agent_detail_token_type ON token_usage_agent_detail(token_type);