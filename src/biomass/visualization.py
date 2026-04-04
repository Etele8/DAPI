from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image for writing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return image.astype(np.uint8).copy()


def draw_labeled_overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    base = _as_bgr(image)
    overlay = base.copy()
    ids = np.unique(labels)
    for label_id in ids:
        if label_id <= 0:
            continue
        color = (
            int((37 * label_id) % 255),
            int((89 * label_id) % 255),
            int((149 * label_id) % 255),
        )
        overlay[labels == label_id] = color
    return cv2.addWeighted(base, 0.55, overlay, 0.45, 0.0)


def draw_contour_overlay(image: np.ndarray, objects: list[object]) -> np.ndarray:
    canvas = _as_bgr(image)
    for obj in objects:
        raw = np.rint(obj.raw_contour_xy).astype(np.int32).reshape(-1, 1, 2)
        smooth = np.rint(obj.smoothed_contour_xy).astype(np.int32).reshape(-1, 1, 2)
        color = (0, 220, 0) if obj.keep else (0, 0, 255)
        cv2.polylines(canvas, [raw], True, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [smooth], True, color, 1, lineType=cv2.LINE_AA)
    return canvas


def draw_object_ids(image: np.ndarray, objects: list[object]) -> np.ndarray:
    canvas = _as_bgr(image)
    for obj in objects:
        x = int(round(obj.centroid[0]))
        y = int(round(obj.centroid[1]))
        color = (0, 220, 0) if obj.keep else (0, 0, 255)
        cv2.putText(canvas, str(obj.object_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(obj.object_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def render_object_panel(image: np.ndarray, obj: object, margin: int = 12) -> np.ndarray:
    x, y, w, h = obj.bbox
    h_img, w_img = image.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w_img, x + w + margin)
    y1 = min(h_img, y + h + margin)

    crop = _as_bgr(image[y0:y1, x0:x1])
    raw = obj.raw_contour_xy - np.array([x0, y0], dtype=np.float64)
    smooth = obj.smoothed_contour_xy - np.array([x0, y0], dtype=np.float64)

    cv2.polylines(crop, [np.rint(raw).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(crop, [np.rint(smooth).astype(np.int32).reshape(-1, 1, 2)], True, (0, 220, 0), 1, cv2.LINE_AA)

    p0 = tuple(np.rint(np.array(obj.longest_chord_endpoints[0]) - np.array([x0, y0])).astype(int))
    p1 = tuple(np.rint(np.array(obj.longest_chord_endpoints[1]) - np.array([x0, y0])).astype(int))
    cv2.line(crop, p0, p1, (255, 0, 0), 1, cv2.LINE_AA)

    text_lines = [
        f"id={obj.object_id} keep={obj.keep}",
        f"area={obj.area_px2_smooth:.1f}px2",
        f"perim={obj.perimeter_px:.1f}px chord={obj.longest_chord_px:.1f}px",
        f"Vz={obj.volume_px3_zeder:.1f} Vr={obj.volume_px3_rod:.1f} Vb={obj.volume_px3_baseline:.1f}",
    ]
    if obj.qc_flags:
        text_lines.append("qc=" + ";".join(obj.qc_flags[:3]))

    y_text = 14
    for line in text_lines:
        cv2.putText(crop, line, (4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(crop, line, (4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += 14
    return crop
