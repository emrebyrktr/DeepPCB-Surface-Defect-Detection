from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, blob: np.ndarray) -> Any:
        pass

    @abstractmethod
    def postprocess(self, outputs: Any, original_shape: tuple) -> dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict[str, Any]:
        pass
