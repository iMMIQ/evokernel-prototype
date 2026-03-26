def update_q_value(current: float, reward: float, alpha: float) -> float:
    return current + alpha * (reward - current)
