"""训练指标工具。"""

from __future__ import annotations

import numpy as np


def accuracy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == targets))
