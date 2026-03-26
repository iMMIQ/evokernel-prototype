from evokernel.domain.enums import Stage
from evokernel.retrieval.reward import update_q_value


QKey = tuple[str, str]


class QValueStore:
    def __init__(self) -> None:
        self.q1: dict[QKey, float] = {}
        self.q2: dict[QKey, float] = {}

    def get(self, stage: Stage, state_signature: str, memory_id: str) -> float:
        return self._bucket(stage).get((state_signature, memory_id), 0.0)

    def update(
        self,
        stage: Stage,
        state_signature: str,
        memory_id: str,
        reward: float,
        alpha: float,
    ) -> float:
        bucket = self._bucket(stage)
        key = (state_signature, memory_id)
        current = bucket.get(key, 0.0)
        bucket[key] = update_q_value(current=current, reward=reward, alpha=alpha)
        return bucket[key]

    def _bucket(self, stage: Stage) -> dict[QKey, float]:
        if stage is Stage.DRAFTING:
            return self.q1
        return self.q2
