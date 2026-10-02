from typing import Protocol

from ..types import BehaviorResult, InteractionFeatures


class BehaviorClassifier(Protocol):
    def classify(self, features: InteractionFeatures, timestamp_s: float) -> BehaviorResult: ...
