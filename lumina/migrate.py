import hashlib
from pathlib import Path


def migrate(dsn: str, directory: Path | None = None) -> list[str]:
    import psycopg

    directory = directory or Path(__file__).with_name("migrations")
    applied = []
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(72134819)")
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            version text PRIMARY KEY, checksum text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now())""")
        for path in sorted(directory.glob("*.sql")):
            content = path.read_text()
            checksum = hashlib.sha256(content.encode()).hexdigest()
            row = conn.execute(
                "SELECT checksum FROM schema_migrations WHERE version = %s", (path.name,)
            ).fetchone()
            if row:
                if row[0] != checksum:
                    raise ValueError(f"Applied migration changed: {path.name}")
                continue
            conn.execute(content)
            conn.execute("INSERT INTO schema_migrations VALUES (%s, %s, now())",
                         (path.name, checksum))
            applied.append(path.name)
    return applied
