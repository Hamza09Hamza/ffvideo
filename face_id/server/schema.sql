-- Face ID recognition schema.
--
-- Scale target: ~200-300 employees, a handful of embeddings each (captured
-- from different angles/lighting during enrollment) — so a few thousand
-- rows in face_embeddings at most. At that size a plain sequential scan
-- with pgvector's cosine distance operator is fast and exact; there's no
-- need for an approximate index (ivfflat/hnsw) until this is much bigger.

CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id SERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    -- ArcFace (buffalo_l / w600k_r50) embeddings are 512-dimensional.
    embedding VECTOR(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS face_embeddings_employee_id_idx ON face_embeddings(employee_id);
