from __future__ import annotations

from ..types import HandObservation, PersonObservation, Point2D


class PersonSelector:
    def __init__(self, max_hand_distance_px: float = 120.0) -> None:
        self.max_hand_distance_px = max_hand_distance_px

    def select(self, people: list[PersonObservation], coaster: Point2D | None) -> PersonObservation | None:
        if coaster is None:
            return None
        return min(people, key=lambda p: min([p.center.distance_to(coaster)] + [a.wrist.distance_to(coaster) for a in p.arms]), default=None)

    def match_hands(self, person: PersonObservation, hands: list[HandObservation]) -> dict[str, HandObservation]:
        matches = {}
        remaining = list(hands)
        for arm in person.arms:
            hand = min(remaining, key=lambda h: h.center.distance_to(arm.wrist), default=None)
            if hand and hand.center.distance_to(arm.wrist) <= self.max_hand_distance_px:
                matches[arm.side] = hand
                remaining.remove(hand)
        return matches
