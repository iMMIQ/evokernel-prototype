import json
import os
import tempfile
from pathlib import Path

from evokernel.domain.models import MemoryItem


class InMemoryStore:
    def __init__(self) -> None:
        self._items: list[MemoryItem] = []
        self._start_points: dict[str, list[MemoryItem]] = {}

    def add(self, item: MemoryItem) -> MemoryItem:
        self._items.append(item)
        if item.became_start_point:
            self._start_points.setdefault(item.task_id, []).append(item)
        return item

    def recall(
        self,
        task_id: str,
        *,
        memory_kind: str | None = None,
        is_feasible: bool | None = None,
    ) -> list[MemoryItem]:
        items = [item for item in self._items if item.task_id == task_id]
        if memory_kind is not None:
            items = [item for item in items if item.memory_kind == memory_kind]
        if is_feasible is not None:
            items = [item for item in items if item.is_feasible is is_feasible]
        return items

    def list_start_points(self, task_id: str) -> list[MemoryItem]:
        return list(self._start_points.get(task_id, []))

    def save_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f"{target.name}.",
            suffix=".tmp",
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for item in self._items:
                    handle.write(json.dumps(item.model_dump(mode="json")))
                    handle.write("\n")
            temp_path.replace(target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

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
