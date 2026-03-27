from __future__ import annotations

import sqlite3
from pathlib import Path

from evokernel.domain.enums import Stage
from evokernel.retrieval.reward import update_q_value


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS q_values (
            stage TEXT NOT NULL,
            state_signature TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (stage, state_signature, memory_id)
        )
        """
    )
    connection.commit()


class QValueStore:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is None:
            connection = sqlite3.connect(":memory:" if db_path is None else Path(db_path))
            connection.row_factory = sqlite3.Row
            self._owns_connection = True
        else:
            self._owns_connection = False
        self._connection = connection
        _initialize_schema(self._connection)

    def get(self, stage: Stage, state_signature: str, memory_id: str) -> float:
        row = self._connection.execute(
            """
            SELECT value
            FROM q_values
            WHERE stage = ? AND state_signature = ? AND memory_id = ?
            """,
            (self._stage_key(stage), state_signature, memory_id),
        ).fetchone()
        if row is None:
            return 0.0
        return float(row["value"])

    def set(self, stage: Stage, state_signature: str, memory_id: str, value: float) -> float:
        self._connection.execute(
            """
            INSERT INTO q_values (stage, state_signature, memory_id, value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(stage, state_signature, memory_id)
            DO UPDATE SET value = excluded.value
            """,
            (self._stage_key(stage), state_signature, memory_id, value),
        )
        self._connection.commit()
        return value

    def update(
        self,
        stage: Stage,
        state_signature: str,
        memory_id: str,
        reward: float,
        alpha: float,
    ) -> float:
        current = self.get(
            stage=stage,
            state_signature=state_signature,
            memory_id=memory_id,
        )
        updated = update_q_value(current=current, reward=reward, alpha=alpha)
        self.set(
            stage=stage,
            state_signature=state_signature,
            memory_id=memory_id,
            value=updated,
        )
        return updated

    def close(self) -> None:
        if self._owns_connection:
            self._connection.close()

    def _stage_key(self, stage: Stage) -> str:
        if stage == Stage.DRAFTING:
            return Stage.DRAFTING.value
        if stage == Stage.REFINING:
            return Stage.REFINING.value
        raise ValueError(f"Unsupported stage: {stage!r}")
