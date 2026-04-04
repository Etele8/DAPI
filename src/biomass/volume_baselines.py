from __future__ import annotations

import math


def equivalent_sphere_volume_from_area(area_px2: float) -> float:
    if area_px2 <= 0:
        return 0.0
    radius = math.sqrt(area_px2 / math.pi)
    return float((4.0 / 3.0) * math.pi * radius**3)


def rod_volume_from_area_and_length(area_px2: float, length_px: float) -> float:
    if area_px2 <= 0 or length_px <= 0:
        return 0.0

    a = math.pi - 4.0
    b = 2.0 * length_px
    c = -area_px2

    if abs(a) < 1e-9:
        radius = area_px2 / max(b, 1e-9)
    else:
        discriminant = max(0.0, b * b - 4.0 * a * c)
        roots = [
            (-b + math.sqrt(discriminant)) / (2.0 * a),
            (-b - math.sqrt(discriminant)) / (2.0 * a),
        ]
        positive_roots = [root for root in roots if root > 0]
        radius = min(positive_roots, default=0.0)

    if radius <= 0:
        return equivalent_sphere_volume_from_area(area_px2)

    cylinder_length = max(0.0, length_px - 2.0 * radius)
    return float(math.pi * radius * radius * cylinder_length + (4.0 / 3.0) * math.pi * radius**3)
