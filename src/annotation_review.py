from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv

import cv2
import numpy as np

from src.config import AnnotationReviewConfig


MANIFEST_FIELDNAMES = ["crop_id", "image_id", "candidate_id", "crop_path", "overlay_path", "label", "notes"]


@dataclass(slots=True)
class AnnotationSessionResult:
    manifest_path: Path
    rows_total: int
    rows_in_scope: int
    rows_labeled: int
    remaining_unlabeled: int
    stopped_early: bool


@dataclass(slots=True)
class AnnotationRunResult:
    manifests: list[AnnotationSessionResult]

    @property
    def manifests_processed(self) -> int:
        return len(self.manifests)

    @property
    def rows_labeled(self) -> int:
        return sum(item.rows_labeled for item in self.manifests)

    @property
    def remaining_unlabeled(self) -> int:
        return sum(item.remaining_unlabeled for item in self.manifests)

    @property
    def stopped_early(self) -> bool:
        return any(item.stopped_early for item in self.manifests)


def _manifest_root(manifest_path: Path) -> Path:
    return manifest_path.parent.parent if manifest_path.parent.name == "annotation" else manifest_path.parent


def _read_image(path: Path) -> np.ndarray:
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _iter_candidates(root_dir: Path, patterns: Iterable[str], recursive: bool) -> list[Path]:
    items: list[Path] = []
    for pattern in patterns:
        if recursive:
            items.extend(root_dir.rglob(pattern))
        else:
            items.extend(root_dir.glob(pattern))
    return sorted({path for path in items if path.is_file()})


def collect_annotation_manifests(root_dir: str | Path, *, manifest_name: str, recursive: bool) -> list[Path]:
    root = Path(root_dir)
    return _iter_candidates(root, [manifest_name], recursive)


def collect_image_files(root_dir: str | Path, *, image_extensions: tuple[str, ...], recursive: bool) -> list[Path]:
    root = Path(root_dir)
    paths = _iter_candidates(root, ["*"], recursive)
    allowed = {ext.lower() for ext in image_extensions}
    return [path for path in paths if path.suffix.lower() in allowed]


def load_annotation_rows(manifest_path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(manifest_path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or MANIFEST_FIELDNAMES)


def save_annotation_rows(manifest_path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with Path(manifest_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_annotation_manifest_from_images(
    root_dir: str | Path,
    *,
    manifest_path: str | Path,
    config: AnnotationReviewConfig,
) -> Path:
    root = Path(root_dir)
    manifest = Path(manifest_path)
    images = collect_image_files(root, image_extensions=config.image_extensions, recursive=config.recursive)
    rows: list[dict[str, str]] = []

    for index, image_path in enumerate(images, start=1):
        relative = image_path.relative_to(root).as_posix()
        stem = image_path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        rows.append(
            {
                "crop_id": stem,
                "image_id": image_path.parent.name or image_path.stem,
                "candidate_id": str(index),
                "crop_path": relative,
                "overlay_path": relative,
                "label": "",
                "notes": "",
            }
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    save_annotation_rows(manifest, rows, MANIFEST_FIELDNAMES)
    return manifest


def ensure_annotation_manifests(
    root_dir: str | Path,
    *,
    config: AnnotationReviewConfig,
) -> list[Path]:
    root = Path(root_dir)
    manifests = collect_annotation_manifests(root, manifest_name=config.manifest_name, recursive=config.recursive)
    if manifests:
        return manifests

    manifest_path = root / config.manifest_name
    if manifest_path.exists():
        return [manifest_path]

    if not collect_image_files(root, image_extensions=config.image_extensions, recursive=config.recursive):
        raise FileNotFoundError(f"No annotation manifests or supported images were found under: {root}")
    return [build_annotation_manifest_from_images(root, manifest_path=manifest_path, config=config)]


def _resolve_variant_paths(
    manifest_path: Path,
    row: dict[str, str],
    config: AnnotationReviewConfig,
) -> list[tuple[str, Path]]:
    manifest_root = _manifest_root(manifest_path)
    field_order = [config.primary_image_field, *config.fallback_image_fields]
    variants: list[tuple[str, Path]] = []
    seen_fields: set[str] = set()
    for field_name in field_order:
        if field_name in seen_fields:
            continue
        seen_fields.add(field_name)
        raw_value = row.get(field_name, "").strip()
        if not raw_value:
            continue
        candidate = manifest_root / raw_value
        if candidate.exists():
            variants.append((field_name, candidate))
    if not variants:
        crop_id = row.get("crop_id", "(unknown)")
        raise FileNotFoundError(f"No readable image path found for {crop_id} in {manifest_path}")
    return variants


def _fit_image(image: np.ndarray, config: AnnotationReviewConfig) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot display an empty image")

    scale_w = config.display_max_width_px / float(w)
    scale_h = config.display_max_height_px / float(h)
    scale = min(scale_w, scale_h, config.max_upscale)
    if scale <= 0:
        scale = 1.0
    if abs(scale - 1.0) < 1e-6:
        return image
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    new_size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    return cv2.resize(image, new_size, interpolation=interpolation)


def _render_frame(
    image: np.ndarray,
    *,
    row: dict[str, str],
    manifest_path: Path,
    view_name: str,
    session_index: int,
    session_total: int,
    config: AnnotationReviewConfig,
) -> np.ndarray:
    display = _fit_image(_as_bgr(image), config)
    text_lines = [
        f"{manifest_path.name}  {session_index}/{session_total}  {row.get('crop_id', '')}  view={view_name}",
        f"label={row.get('label', '').strip() or '(blank)'}  image={row.get('image_id', '')}  candidate={row.get('candidate_id', '')}",
        f"[{config.positive_key.upper()}] {config.positive_label}  "
        f"[{config.negative_key.upper()}] {config.negative_label}  "
        f"[{config.clear_key.upper()}] clear  "
        f"[{config.skip_key.upper()}] skip  "
        f"[{config.back_key.upper()}] back  "
        f"[{config.toggle_view_key.upper()}] view  "
        f"[{config.quit_key.upper()}] quit",
    ]

    line_height = 24
    header_height = 16 + line_height * len(text_lines)
    canvas = np.full((display.shape[0] + header_height, max(display.shape[1], 900), 3), 18, dtype=np.uint8)
    x_offset = max((canvas.shape[1] - display.shape[1]) // 2, 0)
    canvas[header_height : header_height + display.shape[0], x_offset : x_offset + display.shape[1]] = display

    for idx, line in enumerate(text_lines):
        y = 24 + idx * line_height
        cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def _scope_indices(
    rows: list[dict[str, str]],
    *,
    include_labeled: bool,
    start_from_first_unlabeled: bool,
) -> tuple[list[int], int]:
    if include_labeled:
        indices = list(range(len(rows)))
    else:
        indices = [idx for idx, row in enumerate(rows) if not row.get("label", "").strip()]
    if not indices:
        return [], 0

    if include_labeled or not start_from_first_unlabeled:
        return indices, 0
    return indices, 0


def _count_unlabeled(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if not row.get("label", "").strip())


def annotate_manifest(
    manifest_path: str | Path,
    *,
    config: AnnotationReviewConfig,
    include_labeled: bool = False,
) -> AnnotationSessionResult:
    path = Path(manifest_path)
    rows, fieldnames = load_annotation_rows(path)
    indices, pointer = _scope_indices(
        rows,
        include_labeled=include_labeled,
        start_from_first_unlabeled=config.start_from_first_unlabeled,
    )
    if not indices:
        return AnnotationSessionResult(
            manifest_path=path,
            rows_total=len(rows),
            rows_in_scope=0,
            rows_labeled=0,
            remaining_unlabeled=0,
            stopped_early=False,
        )

    if config.fullscreen:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)

    changed_labels = 0
    stopped_early = False
    try:
        while 0 <= pointer < len(indices):
            row_index = indices[pointer]
            row = rows[row_index]
            variants = _resolve_variant_paths(path, row, config)
            view_index = 0

            while True:
                view_name, image_path = variants[view_index]
                image = _read_image(image_path)
                frame = _render_frame(
                    image,
                    row=row,
                    manifest_path=path,
                    view_name=view_name,
                    session_index=pointer + 1,
                    session_total=len(indices),
                    config=config,
                )
                cv2.imshow(config.window_name, frame)
                key = cv2.waitKeyEx(0)
                if key < 0:
                    continue
                char = chr(key & 0xFF).lower() if (key & 0xFF) < 256 else ""

                if key == 27 or char == config.quit_key.lower():
                    stopped_early = True
                    save_annotation_rows(path, rows, fieldnames)
                    return AnnotationSessionResult(
                        manifest_path=path,
                        rows_total=len(rows),
                        rows_in_scope=len(indices),
                        rows_labeled=changed_labels,
                        remaining_unlabeled=_count_unlabeled(rows),
                        stopped_early=stopped_early,
                    )
                if char == config.toggle_view_key.lower() and len(variants) > 1:
                    view_index = (view_index + 1) % len(variants)
                    continue
                if char == config.back_key.lower():
                    pointer = max(pointer - 1, 0)
                    break
                if char == config.skip_key.lower():
                    pointer += 1
                    break
                if char == config.clear_key.lower():
                    previous = row.get("label", "")
                    row["label"] = ""
                    if previous.strip():
                        changed_labels += 1
                    save_annotation_rows(path, rows, fieldnames)
                    pointer += 1
                    break
                if char == config.positive_key.lower():
                    previous = row.get("label", "")
                    row["label"] = config.positive_label
                    if previous.strip() != config.positive_label:
                        changed_labels += 1
                    save_annotation_rows(path, rows, fieldnames)
                    pointer += 1
                    break
                if char == config.negative_key.lower():
                    previous = row.get("label", "")
                    row["label"] = config.negative_label
                    if previous.strip() != config.negative_label:
                        changed_labels += 1
                    save_annotation_rows(path, rows, fieldnames)
                    pointer += 1
                    break
    finally:
        cv2.destroyWindow(config.window_name)

    return AnnotationSessionResult(
        manifest_path=path,
        rows_total=len(rows),
        rows_in_scope=len(indices),
        rows_labeled=changed_labels,
        remaining_unlabeled=_count_unlabeled(rows),
        stopped_early=stopped_early,
    )


def annotate_root(
    root_dir: str | Path,
    *,
    config: AnnotationReviewConfig,
    include_labeled: bool = False,
) -> AnnotationRunResult:
    manifests = ensure_annotation_manifests(root_dir, config=config)
    results: list[AnnotationSessionResult] = []
    for manifest_path in manifests:
        session = annotate_manifest(manifest_path, config=config, include_labeled=include_labeled)
        results.append(session)
        if session.stopped_early:
            break
    return AnnotationRunResult(results)
