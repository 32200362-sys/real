from __future__ import annotations

from ..config import RiskConfig, StateConfig
from ..types import BehaviorLabel, BehaviorResult, MotionFeatures, RiskResult, RobotState, StateUpdate


class StateMachine:
    def __init__(self, state_config: StateConfig, risk_config: RiskConfig) -> None:
        self.config = state_config
        self.risk_config = risk_config
        self.state = RobotState.IDLE
        self._avoid_count = 0
        self._safe_count = 0
        self._hold_candidate_since: float | None = None
        self._return_started_at: float | None = None

    def update(
        self,
        risk: RiskResult,
        feature: MotionFeatures | None,
        target_present: bool,
        timestamp_s: float,
    ) -> StateUpdate:
        if not target_present:
            self._reset_counters()
            self.state = RobotState.IDLE
            return StateUpdate(self.state, "컵/코스터 마커 없음")

        if self.state == RobotState.HOLD:
            if feature is None or feature.distance_to_target_px > self.config.hold_release_distance_px:
                self.state = RobotState.TRACKING
                self._hold_candidate_since = None
                return StateUpdate(self.state, "손이 컵에서 멀어짐")
            return StateUpdate(self.state, "컵을 잡는 동작으로 추정")

        if feature is None:
            self._hold_candidate_since = None
            if self.state == RobotState.AVOIDING:
                self._start_return(timestamp_s)
                return StateUpdate(self.state, "위험 손을 놓쳐 기준점 복귀")
            if self.state == RobotState.RETURNING:
                if self._return_timed_out(timestamp_s):
                    self.state = RobotState.IDLE
                    return StateUpdate(self.state, "복귀 시간 종료")
                return StateUpdate(self.state, "기준점 복귀 중")
            self.state = RobotState.IDLE
            return StateUpdate(self.state, "손 없음")

        hold_condition = (
            feature.distance_to_target_px <= self.risk_config.hold_distance_px
            and feature.speed_px_s <= self.risk_config.hold_speed_px_s
        )
        if hold_condition:
            if self._hold_candidate_since is None:
                self._hold_candidate_since = timestamp_s
            elif timestamp_s - self._hold_candidate_since >= self.config.hold_confirm_seconds:
                self._reset_counters(keep_hold=True)
                self.state = RobotState.HOLD
                return StateUpdate(self.state, "근거리 저속 손 동작 지속")
        else:
            self._hold_candidate_since = None

        if risk.score >= self.risk_config.avoid_threshold:
            self._avoid_count += 1
        else:
            self._avoid_count = 0

        if self._avoid_count >= self.config.avoid_confirm_frames:
            self.state = RobotState.AVOIDING
            self._safe_count = 0
            self._return_started_at = None
            return StateUpdate(self.state, "충돌 위험 확정")

        if self.state == RobotState.AVOIDING:
            if risk.score < self.risk_config.warning_threshold:
                self._safe_count += 1
            else:
                self._safe_count = 0
            if self._safe_count >= self.config.safe_confirm_frames:
                self._start_return(timestamp_s)
                return StateUpdate(self.state, "위험 해제 후 기준점 복귀")
            return StateUpdate(self.state, "회피 동작 유지")

        if self.state == RobotState.RETURNING:
            if risk.score >= self.risk_config.avoid_threshold:
                self.state = RobotState.AVOIDING
                self._return_started_at = None
                return StateUpdate(self.state, "복귀 중 새 위험 감지")
            if self._return_timed_out(timestamp_s):
                self.state = RobotState.TRACKING
                return StateUpdate(self.state, "복귀 시간 종료")
            return StateUpdate(self.state, "기준점 복귀 중")

        if risk.score >= self.risk_config.warning_threshold:
            self.state = RobotState.WARNING
            return StateUpdate(self.state, "접근 위험 주의")

        self.state = RobotState.TRACKING
        return StateUpdate(self.state, "손 추적 중")

    def update_behavior(self, behavior: BehaviorResult, risk: RiskResult,
                        feature: MotionFeatures | None, timestamp_s: float) -> StateUpdate:
        label = behavior.label
        if label == BehaviorLabel.UNKNOWN:
            self.state = RobotState.IDLE
            self._reset_counters()
            return StateUpdate(self.state, f"UNKNOWN: {behavior.reason}")
        if label in {BehaviorLabel.REACHING, BehaviorLabel.HOLDING}:
            self.state = RobotState.HOLD
            self._avoid_count = 0
            return StateUpdate(self.state, behavior.reason)
        if label == BehaviorLabel.COLLISION_RISK:
            return self.update(risk, feature, True, timestamp_s)
        if label == BehaviorLabel.RETRACTING and self.state == RobotState.AVOIDING:
            self._start_return(timestamp_s)
            return StateUpdate(self.state, behavior.reason)
        if self.state == RobotState.RETURNING:
            return self.update(risk, feature, True, timestamp_s)
        self.state = RobotState.TRACKING if feature is not None else RobotState.IDLE
        self._avoid_count = 0
        return StateUpdate(self.state, behavior.reason)

    def force_stop(self) -> StateUpdate:
        self.state = RobotState.IDLE
        self._reset_counters()
        return StateUpdate(self.state, "사용자 강제 정지")

    def _start_return(self, timestamp_s: float) -> None:
        self.state = RobotState.RETURNING
        self._return_started_at = timestamp_s
        self._safe_count = 0
        self._avoid_count = 0

    def _return_timed_out(self, timestamp_s: float) -> bool:
        return (
            self._return_started_at is not None
            and timestamp_s - self._return_started_at >= self.config.return_timeout_seconds
        )

    def _reset_counters(self, keep_hold: bool = False) -> None:
        self._avoid_count = 0
        self._safe_count = 0
        self._return_started_at = None
        if not keep_hold:
            self._hold_candidate_since = None
