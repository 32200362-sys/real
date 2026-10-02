import math


def calculate_ttc(distance_px: float, closing_speed_px_s: float, minimum_speed_px_s: float) -> float | None:
    if not math.isfinite(distance_px) or not math.isfinite(closing_speed_px_s):
        return None
    if distance_px < 0 or closing_speed_px_s < minimum_speed_px_s:
        return None
    value = distance_px / closing_speed_px_s
    return value if math.isfinite(value) else None
