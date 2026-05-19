from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
import tomllib
from typing import Literal


ThresholdMethod = Literal["otsu", "otsu_positive", "percentile", "percentile_positive", "adaptive", "hysteresis"]
DEFAULT_PROFILE_NAME = "default"


@dataclass(slots=True)
class PreprocessingConfig:
    use_median_blur: bool = True
    median_kernel_size: int = 3


@dataclass(slots=True)
class BlueDominanceConfig:
    blue_green_weight: float = 0.6
    blue_red_weight: float = 0.5
    blue_channel_weight: float = 0.25
    local_suppression_kernel_size: int = 35
    local_suppression_strength: float = 0.95


@dataclass(slots=True)
class BlobEnhancementConfig:
    source: Literal["value", "blue", "max_blue_value"] = "max_blue_value"
    top_hat_kernel_size: int = 11
    dog_sigma_small: float = 0.7
    dog_sigma_large: float = 3.0
    top_hat_weight: float = 0.6
    dog_weight: float = 0.8


@dataclass(slots=True)
class LocalSuppressionConfig:
    blur_kernel_size: int = 31
    source: Literal["value", "blue", "max_blue_value"] = "max_blue_value"


@dataclass(slots=True)
class EvidenceFusionConfig:
    blue_weight: float = 0.50
    blob_weight: float = 0.70
    suppression_weight: float = 0.30
    gamma: float = 0.5


@dataclass(slots=True)
class ThresholdConfig:
    method: ThresholdMethod = "hysteresis"
    otsu_scale: float = 1.0
    percentile: float = 95.0
    adaptive_block_size: int = 31
    adaptive_c: float = -4.0
    positive_floor_percentile: float = 85.0
    positive_floor_value: int = 8
    hysteresis_low_scale: float = 0.87
    hysteresis_high_scale: float = 1.05
    pre_blur_kernel_size: int = 1


@dataclass(slots=True)
class MorphologyConfig:
    opening_kernel_size: int = 1
    closing_kernel_size: int = 0
    fill_holes: bool = True
    min_hole_area_px: int = 12


@dataclass(slots=True)
class RegionFilterConfig:
    min_area_px: int = 8
    max_area_px: int | None = 2000
    min_width_px: int = 2
    min_height_px: int = 2
    max_aspect_ratio: float = 3.5
    min_solidity: float = 0.57
    min_convexity: float = 0.85
    min_circularity: float = 0.32
    max_eccentricity: float = 0.95
    exclude_border_touching: bool = False


@dataclass(slots=True)
class SplitConfig:
    enabled: bool = False
    distance_threshold_ratio: float = 0.45
    min_distance_px: int = 2


@dataclass(slots=True)
class SegmentationConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    blue_dominance: BlueDominanceConfig = field(default_factory=BlueDominanceConfig)
    blob_enhancement: BlobEnhancementConfig = field(default_factory=BlobEnhancementConfig)
    local_suppression: LocalSuppressionConfig = field(default_factory=LocalSuppressionConfig)
    fusion: EvidenceFusionConfig = field(default_factory=EvidenceFusionConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    region_filter: RegionFilterConfig = field(default_factory=RegionFilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass(slots=True)
class BiomassConfig:
    microns_per_pixel: float | None = None
    min_area_px2: float = 12.0
    smoothing_points: int = 64
    smoothing_window: int = 7
    min_slice_length_px: float = 0.75
    zeder_max_depth: int = 8
    zeder_width_linearity_tol_px: float = 0.40
    max_debug_panels: int = 24


@dataclass(slots=True)
class ProposalFilterConfig:
    min_area_px: int = 4
    max_area_px: int | None = None
    keep_border_touching: bool = True
    small_area_warn_px: int = 8
    large_area_warn_px: int = 2500
    merged_aspect_ratio_warn: float = 4.5
    merged_eccentricity_warn: float = 0.98
    merged_solidity_warn: float = 0.5


@dataclass(slots=True)
class CropExportConfig:
    enabled: bool = True
    padding_px: int = 24
    min_crop_size_px: int = 72
    force_square: bool = True
    clamp_to_image: bool = True
    export_images: bool = True
    export_overlays: bool = True
    export_masks: bool = True
    images_subdir: str = "crops/images"
    overlays_subdir: str = "crops/overlays"
    masks_subdir: str = "crops/masks"
    metadata_csv_name: str = "crop_metadata.csv"
    metadata_jsonl_name: str = "crop_metadata.jsonl"


@dataclass(slots=True)
class AnnotationExportConfig:
    manifest_subdir: str = "annotation"
    manifest_name: str = "annotation_manifest.csv"
    label_options: tuple[str, ...] = ("single_valid", "invalid", "merged", "uncertain")
    include_helper_columns: bool = True


@dataclass(slots=True)
class AnnotationReviewConfig:
    manifest_name: str = "annotation_manifest.csv"
    recursive: bool = True
    fullscreen: bool = True
    start_from_first_unlabeled: bool = True
    edited_masks_subdir: str = "annotation/edited_masks"
    primary_image_field: str = "overlay_path"
    fallback_image_fields: tuple[str, ...] = ("overlay_path", "crop_path", "mask_path")
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    display_max_width_px: int = 1800
    display_max_height_px: int = 1000
    max_upscale: float = 12.0
    initial_brush_radius_px: int = 6
    min_brush_radius_px: int = 1
    max_brush_radius_px: int = 48
    positive_key: str = "t"
    negative_key: str = "f"
    clear_key: str = "u"
    skip_key: str = "s"
    back_key: str = "b"
    toggle_view_key: str = "v"
    edit_mask_key: str = "m"
    save_mask_key: str = "e"
    reset_mask_key: str = "r"
    undo_key: str = "z"
    decrease_brush_key: str = "1"
    increase_brush_key: str = "2"
    quit_key: str = "q"
    positive_label: str = "single_valid"
    negative_label: str = "invalid"
    window_name: str = "DAPI Annotation"


@dataclass(slots=True)
class ClassifierDatasetConfig:
    positive_labels: tuple[str, ...] = ("single_valid",)
    negative_labels: tuple[str, ...] = ("invalid", "merged")
    excluded_labels: tuple[str, ...] = ("uncertain",)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 13
    allow_image_leakage: bool = False
    split_subdir: str = "classifier_dataset"


def _refinement_segmentation_default() -> SegmentationConfig:
    """Segmentation defaults tuned for local refinement on accepted crops.

    Diverges from the full-image defaults: looser thresholds, larger closing,
    and relaxed region filters so a single in-focus object dominates a crop.
    """
    return SegmentationConfig(
        threshold=ThresholdConfig(
            method="hysteresis",
            otsu_scale=0.95,
            percentile=92.0,
            adaptive_block_size=31,
            adaptive_c=-3.0,
            positive_floor_percentile=70.0,
            positive_floor_value=6,
            hysteresis_low_scale=0.82,
            hysteresis_high_scale=1.0,
            pre_blur_kernel_size=1,
        ),
        morphology=MorphologyConfig(
            opening_kernel_size=1,
            closing_kernel_size=3,
            fill_holes=True,
            min_hole_area_px=18,
        ),
        region_filter=RegionFilterConfig(
            min_area_px=8,
            max_area_px=None,
            min_width_px=2,
            min_height_px=2,
            max_aspect_ratio=8.0,
            min_solidity=0.2,
            min_convexity=0.2,
            min_circularity=0.05,
            max_eccentricity=0.995,
            exclude_border_touching=False,
        ),
    )


@dataclass(slots=True)
class LocalRefinementConfig:
    enabled: bool = True
    segmentation: SegmentationConfig = field(default_factory=_refinement_segmentation_default)
    center_distance_weight: float = 1.0
    refined_masks_subdir: str = "refined/masks"
    refined_overlays_subdir: str = "refined/overlays"
    refined_metadata_name: str = "refined_crops.csv"
    biomass_subdir: str = "refined/biomass"


@dataclass(slots=True)
class ProposalWorkflowConfig:
    filter: ProposalFilterConfig = field(default_factory=ProposalFilterConfig)
    crop_export: CropExportConfig = field(default_factory=CropExportConfig)
    annotation: AnnotationExportConfig = field(default_factory=AnnotationExportConfig)
    review: AnnotationReviewConfig = field(default_factory=AnnotationReviewConfig)
    dataset: ClassifierDatasetConfig = field(default_factory=ClassifierDatasetConfig)
    refinement: LocalRefinementConfig = field(default_factory=LocalRefinementConfig)
    candidates_csv_name: str = "proposal_candidates.csv"
    candidates_jsonl_name: str = "proposal_candidates.jsonl"
    debug_subdir_name: str = "proposal_debug"


@dataclass(slots=True)
class DiscoveryConfig:
    root_dirs: list[Path]
    blue_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    gray_exts: tuple[str, ...] = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    annotated_suffixes: tuple[str, ...] = ("-annotated.jpg", "-annotated.jpeg", "-annotated.png")
    report_txt_suffixes: tuple[str, ...] = ("-report.txt",)
    report_xlsx_suffixes: tuple[str, ...] = ("-report.xlsx",)

    gray_name_markers: tuple[str, ...] = (
        "_grayscale",
        "_greyscale",
        "_grey",
        "_gray",
        "grayscale",
        "greyscale",
        "grey",
        "gray",
    )


@dataclass(slots=True)
class AppConfig:
    discovery: DiscoveryConfig
    prefer_blue: bool = True
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    biomass: BiomassConfig = field(default_factory=BiomassConfig)
    proposal: ProposalWorkflowConfig = field(default_factory=ProposalWorkflowConfig)
    profile_name: str = DEFAULT_PROFILE_NAME
    profile_path: Path | None = None


def _resolve_data_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.exists():
        return path

    raw = str(path_like)
    if raw.startswith(("\\", "/")):
        candidate = Path.cwd() / raw.lstrip("\\/")
        if candidate.exists():
            return candidate

    return path


def _profile_dir() -> Path:
    return Path(__file__).resolve().parent / "profiles"


def available_profiles() -> list[str]:
    profile_dir = _profile_dir()
    if not profile_dir.exists():
        return []
    return sorted(path.stem for path in profile_dir.glob("*.toml"))


def _merge_dataclass_config(base: object, overrides: dict[str, object], *, path: str) -> object:
    if not is_dataclass(base):
        raise TypeError(f"Expected dataclass instance at {path}")

    field_map = {item.name: item for item in fields(base)}
    unknown = sorted(set(overrides) - set(field_map))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown config keys at {path}: {joined}")

    values: dict[str, object] = {}
    for name in field_map:
        current = getattr(base, name)
        if name not in overrides:
            values[name] = current
            continue

        override = overrides[name]
        if is_dataclass(current):
            if not isinstance(override, dict):
                raise ValueError(f"Expected table for {path}.{name}")
            values[name] = _merge_dataclass_config(current, override, path=f"{path}.{name}")
        else:
            values[name] = override

    return type(base)(**values)


def _load_profile_data(*, profile_name: str = DEFAULT_PROFILE_NAME, profile_path: str | Path | None = None) -> tuple[dict[str, object], Path]:
    resolved_profile_path: Path
    if profile_path is not None:
        resolved_profile_path = _resolve_data_root(profile_path)
    else:
        resolved_profile_path = _profile_dir() / f"{profile_name}.toml"

    if not resolved_profile_path.exists():
        if profile_path is not None:
            raise FileNotFoundError(f"Profile file does not exist: {resolved_profile_path}")
        available = ", ".join(available_profiles()) or "(none)"
        raise FileNotFoundError(f"Profile '{profile_name}' was not found. Available profiles: {available}")

    with resolved_profile_path.open("rb") as handle:
        data = tomllib.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Profile must decode to a TOML table: {resolved_profile_path}")
    return data, resolved_profile_path


def apply_profile(
    config: AppConfig,
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
    profile_path: str | Path | None = None,
) -> AppConfig:
    data, resolved_profile_path = _load_profile_data(profile_name=profile_name, profile_path=profile_path)
    allowed_top_level = {"app", "segmentation", "biomass", "proposal"}
    unknown_top_level = sorted(set(data) - allowed_top_level)
    if unknown_top_level:
        joined = ", ".join(unknown_top_level)
        raise ValueError(f"Unknown top-level profile sections: {joined}")

    app_data = data.get("app", {})
    segmentation_data = data.get("segmentation", {})
    biomass_data = data.get("biomass", {})
    proposal_data = data.get("proposal", {})
    if not isinstance(app_data, dict):
        raise ValueError("Profile section 'app' must be a TOML table")
    if not isinstance(segmentation_data, dict):
        raise ValueError("Profile section 'segmentation' must be a TOML table")
    if not isinstance(biomass_data, dict):
        raise ValueError("Profile section 'biomass' must be a TOML table")
    if not isinstance(proposal_data, dict):
        raise ValueError("Profile section 'proposal' must be a TOML table")

    allowed_app_keys = {"prefer_blue"}
    unknown_app_keys = sorted(set(app_data) - allowed_app_keys)
    if unknown_app_keys:
        joined = ", ".join(unknown_app_keys)
        raise ValueError(f"Unknown app config keys: {joined}")

    segmentation = _merge_dataclass_config(config.segmentation, segmentation_data, path="segmentation")
    biomass = _merge_dataclass_config(config.biomass, biomass_data, path="biomass")
    proposal = _merge_dataclass_config(config.proposal, proposal_data, path="proposal")
    prefer_blue = bool(app_data.get("prefer_blue", config.prefer_blue))
    profile_label = resolved_profile_path.stem if profile_path is None else resolved_profile_path.name
    return AppConfig(
        discovery=config.discovery,
        prefer_blue=prefer_blue,
        segmentation=segmentation,
        biomass=biomass,
        proposal=proposal,
        profile_name=profile_label,
        profile_path=resolved_profile_path,
    )


def apply_cli_overrides(config: AppConfig, *, microns_per_pixel: float | None = None) -> AppConfig:
    if microns_per_pixel is None:
        return config
    return replace(config, biomass=replace(config.biomass, microns_per_pixel=microns_per_pixel))


def default_config(
    data_roots: list[str | Path],
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
    profile_path: str | Path | None = None,
) -> AppConfig:
    roots = [_resolve_data_root(p) for p in data_roots]
    config = AppConfig(discovery=DiscoveryConfig(root_dirs=roots))
    return apply_profile(config, profile_name=profile_name, profile_path=profile_path)
