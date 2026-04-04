from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.io.discover import SampleFiles


@dataclass(slots=True)
class LoadedSample:
    stem: str
    source_path: Path
    source_kind: str
    has_blue: bool
    has_gray: bool
    source_image: np.ndarray
    image_bgr_u8: np.ndarray | None
    image_gray_u8: np.ndarray
    annotated_path: Path | None
    report_txt_path: Path | None
    report_xlsx_path: Path | None


def _read_image(path: Path) -> np.ndarray:
    # On Windows, cv2.imread can fail on non-ASCII paths even when the file exists.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _ensure_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    min_v = float(img.min())
    max_v = float(img.max())

    if max_v <= min_v:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_v) / (max_v - min_v)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def _make_brightness_u8(img: np.ndarray) -> np.ndarray:
    """
    Build a brightness image from the per-pixel max across BGR channels.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected BGR image for brightness conversion, got shape {img.shape}")

    brightness = np.max(img.astype(np.float32), axis=2)
    return _ensure_u8(brightness)


def load_sample(sample: SampleFiles, prefer_blue: bool = True) -> LoadedSample:
    # STRICT priority: blue first, gray only fallback
    if sample.blue is not None:
        source_path = sample.blue
        source_kind = "blue_original"
    elif sample.gray is not None:
        source_path = sample.gray
        source_kind = "gray_fallback"
    else:
        raise FileNotFoundError(
            f"No usable image found for sample '{sample.stem}' in folder '{sample.folder}'."
        )

    img = _read_image(source_path)
    image_bgr_u8: np.ndarray | None = None

    # Convert to grayscale properly
    if img.ndim == 2:
        gray = _ensure_u8(img)

    elif img.ndim == 3:
        image_bgr_u8 = _ensure_u8(img)
        if source_kind == "blue_original":
            gray = _make_brightness_u8(image_bgr_u8)
        else:
            # fallback: standard grayscale
            gray = cv2.cvtColor(image_bgr_u8, cv2.COLOR_BGR2GRAY)

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return LoadedSample(
        stem=sample.stem,
        source_path=source_path,
        source_kind=source_kind,
        has_blue=sample.blue is not None,
        has_gray=sample.gray is not None,
        source_image=img,
        image_bgr_u8=image_bgr_u8,
        image_gray_u8=gray,
        annotated_path=sample.annotated,
        report_txt_path=sample.report_txt,
        report_xlsx_path=sample.report_xlsx,
    )
