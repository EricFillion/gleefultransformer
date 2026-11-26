"""Persist workspace jobs and their published models in SQLite."""

import base64
import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .errors import ServiceError


_ACTIVE_STATES = ("running", "cancel_requested")
_PENDING_STATES = ("queued", *_ACTIVE_STATES)
_JSON_FIELDS = ("progress", "metrics", "artifacts", "result", "error")
_MAX_QUEUED_RUNS = 32
_UPDATE_FIELDS = frozenset(("phase", "progress", "metrics", "artifacts"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    sequence INTEGER PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,
    request_id TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    payload TEXT NOT NULL,
    model_id TEXT,
    published_model_id TEXT,
    state TEXT NOT NULL CHECK (state IN (
        'queued', 'running', 'cancel_requested', 'succeeded', 'failed',
        'cancelled', 'interrupted'
    )),
    phase TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    cancel_requested_at TEXT,
    finished_at TEXT,
    worker_id TEXT,
    progress TEXT NOT NULL,
    metrics TEXT NOT NULL DEFAULT '{}',
    artifacts TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    error TEXT,
    UNIQUE (kind, request_id)
);
CREATE INDEX IF NOT EXISTS runs_state_sequence ON runs (state, sequence);
CREATE TABLE IF NOT EXISTS published_models (
    model_id TEXT PRIMARY KEY NOT NULL,
    run_id TEXT UNIQUE NOT NULL REFERENCES runs (run_id),
    record TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _snapshot(row: sqlite3.Row) -> dict:
    snapshot = {
        field: row[field]
        for field in (
            "run_id", "kind", "state", "phase", "model_id", "published_model_id",
            "created_at", "started_at", "updated_at",
            "cancel_requested_at", "finished_at",
        )
    }
    for field in _JSON_FIELDS:
        snapshot[field] = json.loads(row[field]) if row[field] is not None else None
    snapshot["poll_after_ms"] = 500 if row["state"] in _PENDING_STATES else None
    return snapshot


def _request_row(
    connection: sqlite3.Connection, kind: str, request_id: str, request_digest: str
) -> sqlite3.Row | None:
    row = connection.execute(
        "SELECT * FROM runs WHERE kind = ? AND request_id = ?", (kind, request_id)
    ).fetchone()
    if row is not None and row["request_digest"] != request_digest:
        raise ServiceError(
            "request_id_conflict", "This request ID already refers to a different request."
        )
    return row


def _run_row(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ServiceError("run_not_found", "No run has that ID in this workspace.")
    return row


def _owned_run(connection: sqlite3.Connection, run_id: str, worker_id: str) -> sqlite3.Row:
    row = _run_row(connection, run_id)
    if row["worker_id"] != worker_id or row["state"] not in _ACTIVE_STATES:
        raise ServiceError("run_not_owned", "The worker no longer owns an active run with this ID.")
    return row


def _cursor_for(sequence: int, scope: str) -> str:
    return base64.urlsafe_b64encode(_json([sequence, scope]).encode("utf-8")).decode("ascii")


def _cursor_sequence(cursor: str, scope: str) -> int:
    try:
        if len(cursor) > 256:
            raise ValueError("Run listing cursor too long.")
        sequence, cursor_scope = json.loads(
            base64.b64decode(cursor, altchars=b"-_", validate=True)
        )
        if type(sequence) is int and 0 < sequence <= 2**63 - 1 and cursor_scope == scope:
            return sequence
    except (TypeError, ValueError, UnicodeError):
        pass
    raise ServiceError("invalid_cursor", "Use a cursor from a run listing with the same filters.")


class RunStore:
    def __init__(self, workspace: Path):
        self.path = Path(workspace).resolve() / ".gleeful" / "runs.sqlite3"
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if self.path.parent.is_symlink() or self.path.is_symlink():
            raise ServiceError("path_outside_workspace", "The run store cannot use symlinks.")
        with closing(sqlite3.connect(self.path, timeout=10, isolation_level=None)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if write:
                connection.execute("BEGIN IMMEDIATE")
            with connection:
                yield connection

    def lookup_request(self, kind: str, request_id: str, request_payload: Mapping) -> dict | None:
        """Find an idempotent request before revalidating its workspace inputs."""
        with self._connect() as connection:
            row = _request_row(connection, kind, request_id, _digest(request_payload))
            return _snapshot(row) if row is not None else None

    def submit(
        self,
        kind: str,
        request_id: str,
        payload: Mapping,
        *,
        idle_only: bool = False,
        request_payload: Mapping | None = None,
    ) -> dict:
        request_digest = _digest(payload if request_payload is None else request_payload)
        with self._connect(write=True) as connection:
            existing = _request_row(connection, kind, request_id, request_digest)
            if existing is not None:
                return _snapshot(existing)
            if idle_only and connection.execute(
                "SELECT 1 FROM runs WHERE state IN ('queued', 'running', 'cancel_requested')"
            ).fetchone():
                raise ServiceError(
                    "device_busy", "The workspace worker already has pending work.", retryable=True
                )

            pending = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE state = 'queued'"
            ).fetchone()[0]
            if pending >= _MAX_QUEUED_RUNS:
                raise ServiceError(
                    "queue_full", "This workspace already has 32 queued runs.", retryable=True
                )

            run_id = f"run_{uuid4().hex}"
            created_at = _now()
            model_id = payload.get("model_id")
            if model_id is None and isinstance(payload.get("recipe"), Mapping):
                model_id = payload["recipe"].get("model_id")
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, kind, request_id, request_digest, payload,
                    model_id, state, phase, created_at, updated_at, progress
                )
                VALUES (?, ?, ?, ?, ?, ?, 'queued', 'queued', ?, ?, ?)
                """,
                (
                    run_id, kind, request_id, request_digest, _json(payload),
                    model_id, created_at, created_at,
                    _json({"current_step": None, "total_steps": None}),
                ),
            )
            return _snapshot(_run_row(connection, run_id))

    def read(self, run_id: str) -> dict:
        with self._connect() as connection:
            return _snapshot(_run_row(connection, run_id))

    def list_runs(
        self,
        kind: str | None = None,
        state: str | None = None,
        model_id: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> dict:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ServiceError("invalid_limit", "Run listing limits must be between 1 and 100.")
        scope = _digest([str(self.path), kind, state, model_id])
        where = []
        values = []
        for field, value in (("kind", kind), ("state", state)):
            if value is not None:
                where.append(f"{field} = ?")
                values.append(value)
        if model_id is not None:
            where.append("(model_id = ? OR published_model_id = ?)")
            values.extend((model_id, model_id))
        if cursor is not None:
            where.append("sequence < ?")
            values.append(_cursor_sequence(cursor, scope))

        query = "SELECT * FROM runs"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY sequence DESC LIMIT ?"
        with self._connect() as connection:
            rows = connection.execute(query, (*values, limit + 1)).fetchall()
        return {
            "items": [_snapshot(row) for row in rows[:limit]],
            "next_cursor": _cursor_for(rows[limit - 1]["sequence"], scope) if len(rows) > limit else None,
        }

    def stop(self, run_id: str) -> dict:
        with self._connect(write=True) as connection:
            row = _run_row(connection, run_id)
            now = _now()
            if row["state"] == "queued":
                connection.execute(
                    """
                    UPDATE runs SET state = 'cancelled', updated_at = ?,
                        cancel_requested_at = ?, finished_at = ?
                    WHERE run_id = ?
                    """,
                    (now, now, now, run_id),
                )
                action = "cancelled_queued"
            elif row["state"] == "running":
                connection.execute(
                    """
                    UPDATE runs SET state = 'cancel_requested', updated_at = ?, cancel_requested_at = ?
                    WHERE run_id = ?
                    """,
                    (now, now, run_id),
                )
                action = "requested"
            elif row["state"] == "cancel_requested":
                action = "already_requested"
            else:
                action = "already_terminal"
            return {**_snapshot(_run_row(connection, run_id)), "stop_action": action}

    def claim(self, worker_id: str) -> dict | None:
        """Claim the oldest queued job when the single worker is idle."""
        with self._connect(write=True) as connection:
            row = connection.execute(
                """
                SELECT run_id FROM runs WHERE state = 'queued'
                    AND NOT EXISTS (
                        SELECT 1 FROM runs WHERE state IN ('running', 'cancel_requested')
                    )
                ORDER BY sequence LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            run_id = row["run_id"]
            now = _now()
            connection.execute(
                """
                UPDATE runs SET state = 'running', phase = 'loading',
                    started_at = ?, updated_at = ?, worker_id = ?
                WHERE run_id = ?
                """,
                (now, now, worker_id, run_id),
            )
            row = _run_row(connection, run_id)
            return {**_snapshot(row), "payload": json.loads(row["payload"])}

    def update(self, run_id: str, worker_id: str, **fields) -> dict:
        """Record worker progress and report any persisted stop request."""
        if fields.keys() - _UPDATE_FIELDS:
            raise ValueError("Worker updates may contain only phase, progress, metrics, and artifacts.")
        record = {
            field: _json(value) if field in _JSON_FIELDS else value
            for field, value in fields.items()
        }
        with self._connect(write=True) as connection:
            _owned_run(connection, run_id, worker_id)
            record["updated_at"] = _now()
            assignments = ", ".join(f"{field} = ?" for field in record)
            connection.execute(
                f"UPDATE runs SET {assignments} WHERE run_id = ?", (*record.values(), run_id)
            )
            return _snapshot(_run_row(connection, run_id))

    def finish(
        self,
        run_id: str,
        worker_id: str,
        *,
        result: Mapping | None = None,
        error: Mapping | None = None,
        model: Mapping | None = None,
    ) -> dict:
        """Settle a worker result and publish any export in the same transaction."""
        with self._connect(write=True) as connection:
            row = _owned_run(connection, run_id, worker_id)
            published_model_id = None
            artifacts = json.loads(row["artifacts"])
            if row["state"] == "cancel_requested":
                state = "cancelled"
                result = error = None
            elif error is not None:
                if model is not None:
                    raise ValueError("A failed run cannot publish a model.")
                state = "failed"
            else:
                state = "succeeded"
                if model is not None:
                    published_model_id = model["model_id"]
                    connection.execute(
                        "INSERT INTO published_models (model_id, run_id, record) VALUES (?, ?, ?)",
                        (published_model_id, run_id, _json(model)),
                    )
                    artifacts["final_model_id"] = published_model_id
            now = _now()
            connection.execute(
                """
                UPDATE runs SET state = ?, result = ?, error = ?, artifacts = ?,
                    published_model_id = ?, updated_at = ?, finished_at = ?
                WHERE run_id = ?
                """,
                (
                    state, _json(result) if result is not None else None,
                    _json(error) if error is not None else None, _json(artifacts),
                    published_model_id, now, now, run_id,
                ),
            )
            return _snapshot(_run_row(connection, run_id))

    def interrupt_abandoned(self, worker_id: str) -> int:
        """Fence abandoned jobs after taking the exclusive workspace worker lock."""
        with self._connect(write=True) as connection:
            now = _now()
            cursor = connection.execute(
                """
                UPDATE runs SET state = 'interrupted', updated_at = ?, finished_at = ?, error = ?
                WHERE state IN ('running', 'cancel_requested') AND worker_id <> ?
                """,
                (
                    now, now,
                    _json({
                        "code": "worker_interrupted",
                        "message": "The previous workspace worker stopped before this run finished.",
                        "retryable": True,
                    }),
                    worker_id,
                ),
            )
            return cursor.rowcount

    def published_models(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record FROM published_models JOIN runs USING (run_id)
                ORDER BY runs.sequence DESC
                """
            ).fetchall()
        return [json.loads(row["record"]) for row in rows]
