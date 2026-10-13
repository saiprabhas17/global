"""Read-only backup of the GMEE Postgres database.

The database is NEVER written to:
  - pg_dump takes its own REPEATABLE READ, READ ONLY snapshot transaction;
  - every session is additionally forced read-only at the server via
    PGOPTIONS="-c default_transaction_read_only=on", so any accidental write
    would be rejected by Postgres itself;
  - the manifest row counts are plain SELECTs in that same read-only mode.

Source URL resolution (first match wins):
  1. --db-url argument
  2. POSTGRES_URL / DATABASE_URL environment variable
  3. POSTGRES_URL / DATABASE_URL in the repo-root .env
SQLAlchemy driver suffixes (+asyncpg, +psycopg2) are stripped automatically.

Usage (from backend/):
  python scripts/backup_db.py
  python scripts/backup_db.py --db-url "postgresql://user:pass@host:5432/gmee"
  python scripts/backup_db.py --out D:/backups

Output (default backend/_backup/, git-ignored):
  gmee_<db>_<UTC ts>.dump   pg_dump custom format  -> restore with pg_restore
  gmee_<db>_<UTC ts>.json   manifest: sha256, size, server version, row counts

Restore example (into an EMPTY database with pgvector available):
  pg_restore --no-owner --no-privileges -d "postgresql://.../gmee_restore" <file>.dump
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_OUT = BACKEND_DIR / "_backup"

# Exact per-table row counts in ONE read-only query (no temp objects created).
ROW_COUNT_SQL = """
SELECT n.nspname || '.' || c.relname,
       (xpath('/row/c/text()',
              query_to_xml(format('SELECT count(*) AS c FROM %I.%I', n.nspname, c.relname),
                           false, true, '')))[1]::text::bigint
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p')
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg_toast%'
ORDER BY 1;
"""


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def resolve_db_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    dotenv = _load_dotenv(REPO_ROOT / ".env")
    for key in ("POSTGRES_URL", "DATABASE_URL"):
        if os.environ.get(key):
            return os.environ[key]
        if dotenv.get(key):
            return dotenv[key]
    sys.exit("ERROR: no database URL. Pass --db-url or set POSTGRES_URL.")


def to_libpq(url: str) -> tuple[str, str | None, str, str]:
    """Return (libpq URL without password, password, host:port label, db name)."""
    parts = urlsplit(url)
    scheme = parts.scheme.split("+", 1)[0]
    if scheme not in ("postgres", "postgresql"):
        sys.exit(f"ERROR: not a Postgres URL (scheme={parts.scheme!r})")

    user = unquote(parts.username or "")
    password = unquote(parts.password) if parts.password else None
    host = parts.hostname or "localhost"
    port = parts.port or 5432
    db = parts.path.lstrip("/") or "postgres"

    # asyncpg uses ?ssl=require; libpq wants ?sslmode=require.
    query = []
    for k, v in parse_qsl(parts.query):
        query.append(("sslmode", v) if k == "ssl" else (k, v))

    netloc = f"{user}@{host}:{port}" if user else f"{host}:{port}"
    libpq = f"postgresql://{netloc}/{db}"
    if query:
        libpq += "?" + urlencode(query)
    return libpq, password, f"{host}:{port}", db


def find_pg_tool(name: str) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    if os.environ.get("PG_BIN"):
        candidate = Path(os.environ["PG_BIN"]) / exe
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    # Windows installer default location, newest major version first.
    hits = sorted(
        glob.glob(rf"C:\Program Files\PostgreSQL\*\bin\{exe}"),
        key=lambda p: int(Path(p).parts[-3]) if Path(p).parts[-3].isdigit() else 0,
        reverse=True,
    )
    if hits:
        return hits[0]
    sys.exit(f"ERROR: {name} not found. Install PostgreSQL client tools or set PG_BIN.")


def readonly_env(password: str | None) -> dict[str, str]:
    env = os.environ.copy()
    # Server-enforced: any INSERT/UPDATE/DELETE/DDL in this session is rejected.
    env["PGOPTIONS"] = (env.get("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()
    env["PGAPPNAME"] = "gmee_readonly_backup"
    if password is not None:
        env["PGPASSWORD"] = password  # via env, never on the command line
    return env


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    # Callers inspect returncode to produce actionable errors and clean partial dumps.
    return subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", check=False)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only pg_dump backup of the GMEE database.")
    ap.add_argument("--db-url", help="Postgres URL (defaults to POSTGRES_URL from env/.env)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output dir (default {DEFAULT_OUT})")
    args = ap.parse_args()

    libpq_url, password, target, db = to_libpq(resolve_db_url(args.db_url))
    env = readonly_env(password)
    psql, pg_dump, pg_restore = find_pg_tool("psql"), find_pg_tool("pg_dump"), find_pg_tool("pg_restore")

    print(f"Source : {target}/{db}  (read-only session)")

    # 1. Preflight: prove the session is really read-only before dumping anything.
    pre = run(
        [psql, "-X", "-At", "-v", "ON_ERROR_STOP=1", "-d", libpq_url,
         "-c", "SELECT current_setting('default_transaction_read_only'), current_setting('server_version');"],
        env,
    )
    if pre.returncode != 0:
        sys.exit(f"ERROR: cannot connect to {target}/{db}:\n{pre.stderr.strip()}")
    ro_flag, server_version = pre.stdout.strip().split("|", 1)
    if ro_flag != "on":
        sys.exit("ERROR: server did not accept read-only session option; aborting without dumping.")
    print(f"Server : PostgreSQL {server_version}  default_transaction_read_only={ro_flag}")

    # 2. Row counts (read-only SELECTs).
    counts_res = run([psql, "-X", "-At", "-F", "\t", "-v", "ON_ERROR_STOP=1", "-d", libpq_url, "-c", ROW_COUNT_SQL], env)
    if counts_res.returncode != 0:
        sys.exit(f"ERROR: row count query failed:\n{counts_res.stderr.strip()}")
    row_counts = {
        t: int(c) for t, c in (ln.split("\t") for ln in counts_res.stdout.strip().splitlines() if ln)
    }

    # 3. Dump (pg_dump's own snapshot is REPEATABLE READ, READ ONLY).
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dump_path = args.out / f"gmee_{db}_{stamp}.dump"
    tmp_path = dump_path.with_suffix(".dump.partial")
    print(f"Dumping -> {dump_path}")
    dump = run(
        [pg_dump, "-d", libpq_url, "--format=custom", "--compress=6",
         "--no-owner", "--no-privileges", "--file", str(tmp_path)],
        env,
    )
    if dump.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        sys.exit(f"ERROR: pg_dump failed:\n{dump.stderr.strip()}")
    tmp_path.replace(dump_path)

    # 4. Verify the archive is readable (reads the local file only).
    toc = run([pg_restore, "--list", str(dump_path)], env)
    if toc.returncode != 0:
        sys.exit(f"ERROR: dump written but pg_restore cannot read it:\n{toc.stderr.strip()}")
    toc_entries = sum(1 for ln in toc.stdout.splitlines() if ln and not ln.startswith(";"))

    manifest = {
        "created_at_utc": stamp,
        "source": f"{target}/{db}",
        "server_version": server_version,
        "pg_dump": run([pg_dump, "--version"], env).stdout.strip(),
        "mode": "read-only (default_transaction_read_only=on + pg_dump READ ONLY snapshot)",
        "file": dump_path.name,
        "size_bytes": dump_path.stat().st_size,
        "sha256": sha256_of(dump_path),
        "toc_entries": toc_entries,
        "row_counts": row_counts,
        "note": "row_counts are taken just before the dump snapshot; they can differ "
                "slightly if something else was writing to the database concurrently.",
    }
    manifest_path = dump_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK     : {manifest['size_bytes'] / 1_048_576:.2f} MiB, {toc_entries} TOC entries, "
          f"{len(row_counts)} tables, {sum(row_counts.values())} rows")
    for t, c in row_counts.items():
        print(f"         {t:<45} {c:>10}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
