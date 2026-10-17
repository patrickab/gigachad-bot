-- Last resolved reasoning levels per LiteLLM model id, served instantly while a live
-- lookup (OMP registry, Ollama /api/show) revalidates in the background.
-- Model capabilities are global facts, so rows are shared by every user.
CREATE TABLE model_reasoning_levels (
    model text PRIMARY KEY,
    levels text[] NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
