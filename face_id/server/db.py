"""
Embeddings database: enrolled employees + their face embeddings, backed by
Postgres + pgvector.

Scale target is ~200-300 employees with a handful of embeddings each (see
schema.sql) — small enough that an exact nearest-neighbor scan via
pgvector's cosine distance operator (`<=>`) is fast, no approximate index
needed.
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_pool = ConnectionPool(
    conninfo=os.environ["DATABASE_URL"],
    min_size=1,
    max_size=5,
    configure=register_vector,  # teaches psycopg how to send/receive numpy arrays as VECTOR(512)
)


def create_employee(full_name, email=None):
    """Enroll a new employee (without any embeddings yet). Returns the new employee_id."""
    with _pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO employees (full_name, email) VALUES (%s, %s) RETURNING id",
            (full_name, email),
        ).fetchone()
        return row[0]


def add_embedding(employee_id, embedding: np.ndarray):
    """Attach one more captured embedding to an existing employee."""
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO face_embeddings (employee_id, embedding) VALUES (%s, %s)",
            (employee_id, embedding),
        )


def find_best_match(embedding: np.ndarray, threshold: float = 0.5):
    """
    Find the closest enrolled employee to a live embedding.

    pgvector's `<=>` operator is cosine DISTANCE (0 = identical, 2 =
    opposite); we convert to cosine SIMILARITY (1 - distance) so higher
    always means "more alike", matching how the rest of the app reasons
    about scores.

    Returns {"employee_id", "full_name", "similarity"} for the closest
    match if it clears `threshold`, otherwise None (no confident match).
    """
    with _pool.connection() as conn:
        row = conn.execute(
            """
            SELECT e.id, e.full_name, 1 - (fe.embedding <=> %s) AS similarity
            FROM face_embeddings fe
            JOIN employees e ON e.id = fe.employee_id
            ORDER BY fe.embedding <=> %s
            LIMIT 1
            """,
            (embedding, embedding),
        ).fetchone()

    if row is None or row[2] < threshold:
        return None
    return {"employee_id": row[0], "full_name": row[1], "similarity": row[2]}
