from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    classes: list[str]
    model_name: str
    model_version: str
    device: str

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        pass

    @abstractmethod
    def forward(self, blob: np.ndarray) -> Any:
        pass

    @abstractmethod
    def postprocess(
        self,
        outputs: Any,
        orig_shape: tuple[int, int],
        scale_info: tuple[float, float, float],
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict[str, Any]:
        pass
