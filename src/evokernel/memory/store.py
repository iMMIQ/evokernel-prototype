from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from evokernel.domain.models import MemoryItem
from evokernel.memory.document import build_memory_document
from evokernel.memory.embedding import HashingTextEmbedder, TextEmbedder


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_items (
            memory_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            backend_id TEXT NOT NULL,
            operator_family TEXT NOT NULL,
            stage TEXT NOT NULL,
            code TEXT NOT NULL,
            summary TEXT NOT NULL,
            context_summary TEXT,
            memory_kind TEXT NOT NULL,
            reward REAL NOT NULL,
            is_feasible INTEGER NOT NULL,
            became_start_point INTEGER NOT NULL,
            verifier_outcome_json TEXT NOT NULL,
            parent_attempt_id TEXT,
            parent_memory_id TEXT,
            retrieval_text TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        )
        """
    )
    _ensure_column(
        connection,
        table="memory_items",
        column="parent_memory_id",
        definition="TEXT",
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_lookup
        ON memory_items (backend_id, task_id, became_start_point, is_feasible)
        """
    )
    connection.commit()


def _ensure_column(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    definition: str,
) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if any(str(row[1]) == column for row in rows):
        return
    connection.execute(
        f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
    )


class InMemoryStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        embedder: TextEmbedder | None = None,
        reuse_existing: bool = True,
    ) -> None:
        self._path = Path(path).resolve() if path is not None else None
        self._embedder = embedder or HashingTextEmbedder()
        self._reuse_existing = reuse_existing
        self._connection = self._connect()
        _initialize_schema(self._connection)
        self._initial_memory_ids = set(self._fetch_all_memory_ids())
        self._session_memory_ids: set[str] = set()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def loaded_memory_ids(self) -> list[str]:
        if not self._reuse_existing:
            return []
        return sorted(self._initial_memory_ids)

    def add(self, item: MemoryItem) -> MemoryItem:
        retrieval_text = item.retrieval_text or build_memory_document(item)
        embedding = item.embedding or self._embedder.embed_texts([retrieval_text])[0]
        persisted_item = item.model_copy(
            update={
                "retrieval_text": retrieval_text,
                "embedding": embedding,
            }
        )
        verifier_payload = json.dumps(
            persisted_item.verifier_outcome.model_dump(mode="json"),
            sort_keys=True,
        )
        self._connection.execute(
            """
            INSERT OR REPLACE INTO memory_items (
                memory_id,
                task_id,
                backend_id,
                operator_family,
                stage,
                code,
                summary,
                context_summary,
                memory_kind,
                reward,
                is_feasible,
                became_start_point,
                verifier_outcome_json,
                parent_attempt_id,
                parent_memory_id,
                retrieval_text,
                embedding_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                persisted_item.memory_id,
                persisted_item.task_id,
                persisted_item.backend_id,
                persisted_item.operator_family,
                persisted_item.stage.value,
                persisted_item.code,
                persisted_item.summary,
                persisted_item.context_summary,
                persisted_item.memory_kind,
                persisted_item.reward,
                int(persisted_item.is_feasible),
                int(persisted_item.became_start_point),
                verifier_payload,
                persisted_item.parent_attempt_id,
                persisted_item.parent_memory_id,
                retrieval_text,
                json.dumps(embedding),
            ),
        )
        self._connection.commit()
        self._session_memory_ids.add(persisted_item.memory_id)
        return persisted_item

    def recall(
        self,
        task_id: str | None = None,
        *,
        backend_id: str | None = None,
        operator_family: str | None = None,
        memory_kind: str | None = None,
        is_feasible: bool | None = None,
        became_start_point: bool | None = None,
        parent_memory_id: str | None = None,
        exclude_memory_ids: set[str] | None = None,
    ) -> list[MemoryItem]:
        visible_ids = self._visible_memory_ids()
        if not self._reuse_existing and not visible_ids:
            return []

        clauses: list[str] = []
        parameters: list[object] = []

        if task_id is not None:
            clauses.append("task_id = ?")
            parameters.append(task_id)
        if backend_id is not None:
            clauses.append("backend_id = ?")
            parameters.append(backend_id)
        if operator_family is not None:
            clauses.append("operator_family = ?")
            parameters.append(operator_family)
        if memory_kind is not None:
            clauses.append("memory_kind = ?")
            parameters.append(memory_kind)
        if is_feasible is not None:
            clauses.append("is_feasible = ?")
            parameters.append(int(is_feasible))
        if became_start_point is not None:
            clauses.append("became_start_point = ?")
            parameters.append(int(became_start_point))
        if parent_memory_id is not None:
            clauses.append("parent_memory_id = ?")
            parameters.append(parent_memory_id)

        excluded_ids = set(exclude_memory_ids or ())
        if not self._reuse_existing:
            clauses.append(
                "memory_id IN ("
                + ",".join("?" for _ in visible_ids)
                + ")"
            )
            parameters.extend(sorted(visible_ids))
        elif excluded_ids:
            clauses.append(
                "memory_id NOT IN ("
                + ",".join("?" for _ in excluded_ids)
                + ")"
            )
            parameters.extend(sorted(excluded_ids))

        query = "SELECT * FROM memory_items"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        rows = self._connection.execute(query, parameters).fetchall()
        items = [self._row_to_item(row) for row in rows]
        if excluded_ids:
            items = [
                item for item in items if item.memory_id not in excluded_ids
            ]
        return items

    def list_start_points(self, task_id: str) -> list[MemoryItem]:
        visible_ids = self._visible_memory_ids()
        clauses = [
            "task_id = ?",
            "is_feasible = 1",
            "became_start_point = 1",
        ]
        parameters: list[object] = [task_id]
        if not self._reuse_existing:
            if not visible_ids:
                return []
            clauses.append(
                "memory_id IN ("
                + ",".join("?" for _ in visible_ids)
                + ")"
            )
            parameters.extend(sorted(visible_ids))

        rows = self._connection.execute(
            "SELECT * FROM memory_items WHERE "
            + " AND ".join(clauses)
            + " ORDER BY rowid DESC",
            parameters,
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def save_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f"{target.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for item in self.recall():
                handle.write(json.dumps(item.model_dump(mode="json")))
                handle.write("\n")
        temp_path.replace(target)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "InMemoryStore":
        store = cls()
        source = Path(path)
        if not source.exists():
            return store

        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                store.add(MemoryItem.model_validate(json.loads(payload)))
        return store

    def close(self) -> None:
        self._connection.close()

    def _connect(self) -> sqlite3.Connection:
        if self._path is None:
            connection = sqlite3.connect(":memory:")
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def _fetch_all_memory_ids(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT memory_id FROM memory_items"
        ).fetchall()
        return [str(row["memory_id"]) for row in rows]

    def _visible_memory_ids(self) -> set[str]:
        if self._reuse_existing:
            return set()
        return set(self._session_memory_ids)

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            memory_id=str(row["memory_id"]),
            task_id=str(row["task_id"]),
            backend_id=str(row["backend_id"]),
            operator_family=str(row["operator_family"]),
            stage=str(row["stage"]),
            code=str(row["code"]),
            summary=str(row["summary"]),
            context_summary=row["context_summary"],
            memory_kind=str(row["memory_kind"]),
            reward=float(row["reward"]),
            is_feasible=bool(row["is_feasible"]),
            became_start_point=bool(row["became_start_point"]),
            verifier_outcome=json.loads(str(row["verifier_outcome_json"])),
            parent_attempt_id=row["parent_attempt_id"],
            parent_memory_id=row["parent_memory_id"],
            retrieval_text=str(row["retrieval_text"]),
            embedding=[
                float(component)
                for component in json.loads(str(row["embedding_json"]))
            ],
        )
