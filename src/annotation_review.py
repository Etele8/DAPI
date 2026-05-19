from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import csv

import cv2
import numpy as np

from src.config import AnnotationReviewConfig


MANIFEST_FIELDNAMES = [
    "crop_id",
    "image_id",
    "candidate_id",
    "crop_path",
    "overlay_path",
    "mask_path",
    "edited_mask_path",
    "mask_was_edited",
    "mask_edit_mode",
    "label",
    "notes",
]


@dataclass(slots=True)
class AnnotationSessionResult:
    manifest_path: Path
    rows_total: int
    rows_in_scope: int
    rows_labeled: int
    masks_saved: int
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
    def masks_saved(self) -> int:
        return sum(item.masks_saved for item in self.manifests)

    @property
    def remaining_unlabeled(self) -> int:
        return sum(item.remaining_unlabeled for item in self.manifests)

    @property
    def stopped_early(self) -> bool:
        return any(item.stopped_early for item in self.manifests)


@dataclass(slots=True)
class MaskEditState:
    active: bool = False
    brush_radius_px: int = 6
    min_brush_radius_px: int = 1
    max_brush_radius_px: int = 48
    current_mask: np.ndarray | None = None
    original_mask: np.ndarray | None = None
    undo_stack: list[np.ndarray] = field(default_factory=list)
    mouse_down: bool = False
    draw_value: int = 255
    last_canvas_xy: tuple[int, int] | None = None
    pointer_display_xy: tuple[int, int] | None = None
    display_scale: float = 1.0
    image_offset_xy: tuple[int, int] = (0, 0)
    image_shape_hw: tuple[int, int] = (0, 0)
    dirty: bool = False

    def begin(self, *, base_mask: np.ndarray, config: AnnotationReviewConfig) -> None:
        self.active = True
        self.brush_radius_px = int(np.clip(self.brush_radius_px or config.initial_brush_radius_px, config.min_brush_radius_px, config.max_brush_radius_px))
        self.original_mask = base_mask.copy()
        self.current_mask = base_mask.copy()
        self.undo_stack = [self.current_mask.copy()]
        self.mouse_down = False
        self.last_canvas_xy = None
        self.pointer_display_xy = None
        self.dirty = False

    def has_mask(self) -> bool:
        return self.current_mask is not None

    def reset(self) -> None:
        if self.original_mask is None:
            return
        self.current_mask = self.original_mask.copy()
        self.undo_stack = [self.current_mask.copy()]
        self.mouse_down = False
        self.last_canvas_xy = None
        self.dirty = False

    def save_undo_snapshot(self) -> None:
        if self.current_mask is None:
            return
        if self.undo_stack and np.array_equal(self.undo_stack[-1], self.current_mask):
            return
        self.undo_stack.append(self.current_mask.copy())
        if len(self.undo_stack) > 32:
            self.undo_stack = self.undo_stack[-32:]

    def undo(self) -> None:
        if len(self.undo_stack) <= 1:
            return
        self.undo_stack.pop()
        self.current_mask = self.undo_stack[-1].copy()
        self.mouse_down = False
        self.last_canvas_xy = None
        self.dirty = True


def _manifest_root(manifest_path: Path) -> Path:
    return manifest_path.parent.parent if manifest_path.parent.name == "annotation" else manifest_path.parent


def _read_image(path: Path, *, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image for writing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


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


def _normalize_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    final_fieldnames = list(fieldnames or MANIFEST_FIELDNAMES)
    for name in MANIFEST_FIELDNAMES:
        if name not in final_fieldnames:
            final_fieldnames.append(name)

    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized_row = dict(row)
        for name in final_fieldnames:
            normalized_row.setdefault(name, "")
        normalized.append(normalized_row)
    return normalized, final_fieldnames


def load_annotation_rows(manifest_path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(manifest_path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows, fieldnames = list(reader), list(reader.fieldnames or MANIFEST_FIELDNAMES)
    return _normalize_rows(rows, fieldnames)


def save_annotation_rows(manifest_path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    normalized_rows, normalized_fieldnames = _normalize_rows(rows, fieldnames)
    with Path(manifest_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=normalized_fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


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
                "mask_path": "",
                "edited_mask_path": "",
                "mask_was_edited": "false",
                "mask_edit_mode": "",
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


def _resolve_row_path(manifest_path: Path, row: dict[str, str], field_name: str) -> Path | None:
    manifest_root = _manifest_root(manifest_path)
    raw_value = row.get(field_name, "").strip()
    candidates: list[Path] = []
    if raw_value:
        candidates.append(manifest_root / raw_value)
    if field_name == "mask_path":
        crop_value = row.get("crop_path", "").strip()
        if crop_value:
            candidates.append(manifest_root / crop_value.replace("crops/images/", "crops/masks/"))
            crop_path = Path(crop_value)
            candidates.append(manifest_root / crop_path.parent.parent / "masks" / crop_path.name)
    for path in candidates:
        if path.exists():
            return path
    return None


def _build_display_variants(
    manifest_path: Path,
    row: dict[str, str],
    config: AnnotationReviewConfig,
    *,
    edit_active: bool,
) -> list[tuple[str, Path | None]]:
    variants: list[tuple[str, Path | None]] = []
    if edit_active:
        crop_path = _resolve_row_path(manifest_path, row, "crop_path")
        if crop_path is not None:
            variants.append(("raw_crop", crop_path))
        variants.append(("edited_mask", None))
        return variants

    for field_name, path in _resolve_variant_paths(manifest_path, row, config):
        variants.append((field_name, path))
    return variants


def _load_mask_from_row(
    manifest_path: Path,
    row: dict[str, str],
    *,
    prefer_edited: bool = True,
) -> np.ndarray | None:
    field_order = ("edited_mask_path", "mask_path") if prefer_edited else ("mask_path", "edited_mask_path")
    for field_name in field_order:
        path = _resolve_row_path(manifest_path, row, field_name)
        if path is None:
            continue
        mask = _read_image(path, flags=cv2.IMREAD_GRAYSCALE)
        return np.where(mask > 0, 255, 0).astype(np.uint8)
    return None


def _canonical_crop_shape(manifest_path: Path, row: dict[str, str]) -> tuple[int, int] | None:
    """Shape of the raw crop image. Editing canvases and masks must align to this."""
    crop_path = _resolve_row_path(manifest_path, row, "crop_path")
    if crop_path is None:
        return None
    return _read_image(crop_path).shape[:2]


def _resize_mask_to_shape(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to target_hw, preserving 0/255 values via nearest-neighbour."""
    if mask.shape[:2] == target_hw:
        return mask
    resized = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 255, 0).astype(np.uint8)


def _fit_image(image: np.ndarray, config: AnnotationReviewConfig) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot display an empty image")

    scale_w = config.display_max_width_px / float(w)
    scale_h = config.display_max_height_px / float(h)
    scale = min(scale_w, scale_h, config.max_upscale)
    if scale <= 0:
        scale = 1.0
    if abs(scale - 1.0) < 1e-6:
        return image, 1.0
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    new_size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    return cv2.resize(image, new_size, interpolation=interpolation), scale


def _render_frame(
    image: np.ndarray,
    *,
    row: dict[str, str],
    manifest_path: Path,
    view_name: str,
    session_index: int,
    session_total: int,
    config: AnnotationReviewConfig,
    edit_state: MaskEditState,
) -> np.ndarray:
    display_base = _as_bgr(image)
    display, scale = _fit_image(display_base, config)
    mode_text = (
        f"edit=mask brush={edit_state.brush_radius_px}px dirty={edit_state.dirty}"
        if edit_state.active
        else "edit=off"
    )
    text_lines = [
        f"{manifest_path.name}  {session_index}/{session_total}  {row.get('crop_id', '')}  view={view_name}",
        f"label={row.get('label', '').strip() or '(blank)'}  image={row.get('image_id', '')}  candidate={row.get('candidate_id', '')}",
        mode_text,
        f"[{config.positive_key.upper()}] {config.positive_label}  "
        f"[{config.negative_key.upper()}] {config.negative_label}  "
        f"[{config.clear_key.upper()}] clear  "
        f"[{config.skip_key.upper()}] skip  "
        f"[{config.back_key.upper()}] back  "
        f"[{config.toggle_view_key.upper()}] view",
        f"[{config.edit_mask_key.upper()}] edit mask  "
        f"[{config.save_mask_key.upper()}] save mask  "
        f"[{config.reset_mask_key.upper()}] reset  "
        f"[{config.undo_key.upper()}] undo  "
        f"[{config.decrease_brush_key}] smaller  "
        f"[{config.increase_brush_key}] larger  "
        f"[{config.quit_key.upper()}] quit",
    ]

    line_height = 24
    header_height = 16 + line_height * len(text_lines)
    canvas = np.full((display.shape[0] + header_height, max(display.shape[1], 980), 3), 18, dtype=np.uint8)
    x_offset = max((canvas.shape[1] - display.shape[1]) // 2, 0)
    canvas[header_height : header_height + display.shape[0], x_offset : x_offset + display.shape[1]] = display

    edit_state.display_scale = scale
    edit_state.image_offset_xy = (x_offset, header_height)
    edit_state.image_shape_hw = display_base.shape[:2]

    if edit_state.active and edit_state.pointer_display_xy is not None:
        px, py = edit_state.pointer_display_xy
        radius = max(int(round(edit_state.brush_radius_px * scale)), 1)
        center = (x_offset + px, header_height + py)
        cv2.circle(canvas, center, radius, (255, 255, 0), 1, cv2.LINE_AA)

    for idx, line in enumerate(text_lines):
        y = 24 + idx * line_height
        cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def _scope_indices(
    rows: list[dict[str, str]],
    *,
    include_labeled: bool,
) -> tuple[list[int], int]:
    indices = list(range(len(rows))) if include_labeled else [idx for idx, row in enumerate(rows) if not row.get("label", "").strip()]
    return indices, 0


def _count_unlabeled(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if not row.get("label", "").strip())


def _edited_mask_output_path(manifest_path: Path, row: dict[str, str], config: AnnotationReviewConfig) -> Path:
    manifest_root = _manifest_root(manifest_path)
    crop_id = row.get("crop_id", "").strip() or f"row_{row.get('candidate_id', '0')}"
    return manifest_root / config.edited_masks_subdir / f"{crop_id}.png"


def _save_mask_edit(
    *,
    manifest_path: Path,
    row: dict[str, str],
    config: AnnotationReviewConfig,
    edit_state: MaskEditState,
) -> bool:
    if edit_state.current_mask is None:
        return False
    output_path = _edited_mask_output_path(manifest_path, row, config)
    _write_image(output_path, edit_state.current_mask)
    row["edited_mask_path"] = output_path.relative_to(_manifest_root(manifest_path)).as_posix()
    row["mask_was_edited"] = "true"
    row["mask_edit_mode"] = "edit"
    edit_state.dirty = False
    return True


def _mask_point_from_canvas(x: int, y: int, edit_state: MaskEditState) -> tuple[int, int] | None:
    offset_x, offset_y = edit_state.image_offset_xy
    local_x = x - offset_x
    local_y = y - offset_y
    display_w = int(round(edit_state.image_shape_hw[1] * edit_state.display_scale))
    display_h = int(round(edit_state.image_shape_hw[0] * edit_state.display_scale))
    if local_x < 0 or local_y < 0 or local_x >= display_w or local_y >= display_h:
        return None
    image_x = int(np.clip(round(local_x / edit_state.display_scale), 0, edit_state.image_shape_hw[1] - 1))
    image_y = int(np.clip(round(local_y / edit_state.display_scale), 0, edit_state.image_shape_hw[0] - 1))
    return image_x, image_y


def _paint_line(edit_state: MaskEditState, start_xy: tuple[int, int], end_xy: tuple[int, int]) -> None:
    if edit_state.current_mask is None:
        return
    delta_x = end_xy[0] - start_xy[0]
    delta_y = end_xy[1] - start_xy[1]
    steps = max(abs(delta_x), abs(delta_y), 1)
    for step in range(steps + 1):
        alpha = step / float(steps)
        px = int(round(start_xy[0] + alpha * delta_x))
        py = int(round(start_xy[1] + alpha * delta_y))
        cv2.circle(edit_state.current_mask, (px, py), edit_state.brush_radius_px, int(edit_state.draw_value), thickness=-1, lineType=cv2.LINE_AA)
    edit_state.current_mask = np.where(edit_state.current_mask > 0, 255, 0).astype(np.uint8)
    edit_state.dirty = True


def _annotation_mouse_callback(event: int, x: int, y: int, flags: int, userdata: MaskEditState) -> None:
    state = userdata
    if not state.active or state.current_mask is None:
        return
    point = _mask_point_from_canvas(x, y, state)
    if point is None:
        state.pointer_display_xy = None
        if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            state.mouse_down = False
            state.last_canvas_xy = None
        return

    display_point = (
        int(round(point[0] * state.display_scale)),
        int(round(point[1] * state.display_scale)),
    )
    state.pointer_display_xy = display_point

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        state.save_undo_snapshot()
        state.mouse_down = True
        state.draw_value = 255 if event == cv2.EVENT_LBUTTONDOWN else 0
        state.last_canvas_xy = point
        _paint_line(state, point, point)
        return

    if event == cv2.EVENT_MOUSEMOVE and state.mouse_down:
        if state.last_canvas_xy is None:
            state.last_canvas_xy = point
        _paint_line(state, state.last_canvas_xy, point)
        state.last_canvas_xy = point
        return

    if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        if state.mouse_down and state.last_canvas_xy is not None:
            _paint_line(state, state.last_canvas_xy, point)
        state.mouse_down = False
        state.last_canvas_xy = None


def annotate_manifest(
    manifest_path: str | Path,
    *,
    config: AnnotationReviewConfig,
    include_labeled: bool = False,
) -> AnnotationSessionResult:
    path = Path(manifest_path)
    rows, fieldnames = load_annotation_rows(path)
    indices, pointer = _scope_indices(rows, include_labeled=include_labeled)
    if not indices:
        return AnnotationSessionResult(
            manifest_path=path,
            rows_total=len(rows),
            rows_in_scope=0,
            rows_labeled=0,
            masks_saved=0,
            remaining_unlabeled=_count_unlabeled(rows),
            stopped_early=False,
        )

    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    if config.fullscreen:
        cv2.setWindowProperty(config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    edit_state = MaskEditState(
        brush_radius_px=config.initial_brush_radius_px,
        min_brush_radius_px=config.min_brush_radius_px,
        max_brush_radius_px=config.max_brush_radius_px,
    )
    cv2.setMouseCallback(config.window_name, _annotation_mouse_callback, edit_state)

    changed_labels = 0
    masks_saved = 0
    stopped_early = False
    try:
        while 0 <= pointer < len(indices):
            row_index = indices[pointer]
            row = rows[row_index]
            view_index = 0
            canonical_hw = _canonical_crop_shape(path, row)
            loaded_mask = _load_mask_from_row(path, row, prefer_edited=True)
            original_mask = _load_mask_from_row(path, row, prefer_edited=False)
            if canonical_hw is not None:
                if loaded_mask is not None:
                    loaded_mask = _resize_mask_to_shape(loaded_mask, canonical_hw)
                if original_mask is not None:
                    original_mask = _resize_mask_to_shape(original_mask, canonical_hw)

            if loaded_mask is not None:
                edit_state.original_mask = loaded_mask.copy()
                if edit_state.current_mask is None or not edit_state.active:
                    edit_state.current_mask = loaded_mask.copy()
                edit_state.image_shape_hw = loaded_mask.shape[:2]
            else:
                edit_state.active = False
                edit_state.current_mask = None
                edit_state.original_mask = None
                edit_state.undo_stack = []
                edit_state.dirty = False

            while True:
                variants = _build_display_variants(path, row, config, edit_active=edit_state.active)
                view_index = min(view_index, len(variants) - 1)
                view_name, image_path = variants[view_index]

                if view_name == "edited_mask":
                    source_mask = edit_state.current_mask
                    if source_mask is None:
                        raise ValueError("Edited mask view requested without an active mask")
                    image = source_mask
                else:
                    if image_path is None:
                        raise ValueError(f"Missing image path for view {view_name}")
                    flags = cv2.IMREAD_GRAYSCALE if "mask" in view_name else cv2.IMREAD_UNCHANGED
                    image = _read_image(image_path, flags=flags)

                frame = _render_frame(
                    image,
                    row=row,
                    manifest_path=path,
                    view_name=view_name,
                    session_index=pointer + 1,
                    session_total=len(indices),
                    config=config,
                    edit_state=edit_state,
                )
                cv2.imshow(config.window_name, frame)
                key = cv2.waitKeyEx(30)
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
                        masks_saved=masks_saved,
                        remaining_unlabeled=_count_unlabeled(rows),
                        stopped_early=True,
                    )
                if char == config.toggle_view_key.lower() and len(variants) > 1:
                    view_index = (view_index + 1) % len(variants)
                    continue
                if char == config.decrease_brush_key.lower():
                    edit_state.brush_radius_px = max(edit_state.brush_radius_px - 1, config.min_brush_radius_px)
                    continue
                if char == config.increase_brush_key.lower():
                    edit_state.brush_radius_px = min(edit_state.brush_radius_px + 1, config.max_brush_radius_px)
                    continue
                if char == config.edit_mask_key.lower():
                    base_mask = original_mask if original_mask is not None else loaded_mask
                    if base_mask is None:
                        target_hw = canonical_hw if canonical_hw is not None else image.shape[:2]
                        base_mask = np.zeros(target_hw, dtype=np.uint8)
                    edit_state.begin(base_mask=base_mask, config=config)
                    view_index = 0
                    continue
                if char == config.reset_mask_key.lower() and edit_state.active:
                    edit_state.reset()
                    continue
                if char == config.undo_key.lower() and edit_state.active:
                    edit_state.undo()
                    continue
                if char == config.save_mask_key.lower() and edit_state.active:
                    if _save_mask_edit(manifest_path=path, row=row, config=config, edit_state=edit_state):
                        save_annotation_rows(path, rows, fieldnames)
                        masks_saved += 1
                        loaded_mask = edit_state.current_mask.copy() if edit_state.current_mask is not None else None
                    continue
                if char == config.back_key.lower():
                    pointer = max(pointer - 1, 0)
                    edit_state.mouse_down = False
                    break
                if char == config.skip_key.lower():
                    pointer += 1
                    edit_state.mouse_down = False
                    break
                if char == config.clear_key.lower():
                    previous = row.get("label", "")
                    row["label"] = ""
                    if previous.strip():
                        changed_labels += 1
                    save_annotation_rows(path, rows, fieldnames)
                    pointer += 1
                    edit_state.mouse_down = False
                    break
                if char == config.positive_key.lower():
                    previous = row.get("label", "")
                    row["label"] = config.positive_label
                    if previous.strip() != config.positive_label:
                        changed_labels += 1
                    save_annotation_rows(path, rows, fieldnames)
                    pointer += 1
                    edit_state.mouse_down = False
                    break
                if char == config.negative_key.lower():
                    previous = row.get("label", "")
                    row["label"] = config.negative_label
                    if previous.strip() != config.negative_label:
                        changed_labels += 1
                    save_annotation_rows(path, rows, fieldnames)
                    pointer += 1
                    edit_state.mouse_down = False
                    break
    finally:
        cv2.destroyWindow(config.window_name)

    return AnnotationSessionResult(
        manifest_path=path,
        rows_total=len(rows),
        rows_in_scope=len(indices),
        rows_labeled=changed_labels,
        masks_saved=masks_saved,
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
