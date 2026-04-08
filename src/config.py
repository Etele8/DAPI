from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal


ThresholdMethod = Literal["otsu", "otsu_positive", "percentile", "percentile_positive", "adaptive", "hysteresis"]


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
class DiscoveryConfig:
    root_dirs: list[Path]
    blue_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    gray_exts: tuple[str, ...] = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    annotated_suffixes: tuple[str, ...] = ("-annotated.jpg", "-annotated.jpeg", "-annotated.png")
    report_txt_suffixes: tuple[str, ...] = ("-report.txt",)
    report_xlsx_suffixes: tuple[str, ...] = ("-report.xlsx",)

    # gray variants seen in your dataset
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
    segmentation: SegmentationConfig = field(default_factory=lambda: default_segmentation_config())
    biomass: BiomassConfig = field(default_factory=lambda: default_biomass_config())


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


def default_segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(
        preprocessing=PreprocessingConfig(
            use_median_blur=True,
            median_kernel_size=3,
        ),
        blue_dominance=BlueDominanceConfig(
            blue_green_weight=0.6,
            blue_red_weight=0.5,
            blue_channel_weight=0.25,
            local_suppression_kernel_size=35,
            local_suppression_strength=0.95,
        ),
        blob_enhancement=BlobEnhancementConfig(
            source="max_blue_value",
            top_hat_kernel_size=11,
            dog_sigma_small=0.7,
            dog_sigma_large=3.0,
            top_hat_weight=0.6,
            dog_weight=0.8,
        ),
        local_suppression=LocalSuppressionConfig(
            blur_kernel_size=31,
            source="max_blue_value",
        ),
        fusion=EvidenceFusionConfig(
            blue_weight=0.50,
            blob_weight=0.70,
            suppression_weight=0.30,
            gamma=0.5,
        ),
        threshold=ThresholdConfig(
            method="hysteresis",
            otsu_scale=1.0,
            percentile=95.0,
            adaptive_block_size=31,
            adaptive_c=-4.0,
            positive_floor_percentile=85.0,
            positive_floor_value=8,
            hysteresis_low_scale=0.87,
            hysteresis_high_scale=1.05,
            pre_blur_kernel_size=1,
        ),
        morphology=MorphologyConfig(
            opening_kernel_size=1,
            closing_kernel_size=0,
            fill_holes=True,
            min_hole_area_px=12,
        ),
        region_filter=RegionFilterConfig(
            min_area_px=8,
            max_area_px=2000,
            min_width_px=2,
            min_height_px=2,
            max_aspect_ratio=3.5,
            min_solidity=0.57,
            min_convexity=0.85,
            min_circularity=0.32,
            max_eccentricity=0.95,
            exclude_border_touching=False,
        ),
        split=SplitConfig(
            enabled=False,
            distance_threshold_ratio=0.45,
            min_distance_px=2,
        ),
    )


def default_biomass_config() -> BiomassConfig:
    return BiomassConfig(
        microns_per_pixel=None,
        min_area_px2=12.0,
        smoothing_points=64,
        smoothing_window=7,
        min_slice_length_px=0.75,
        zeder_max_depth=8,
        zeder_width_linearity_tol_px=0.40,
        max_debug_panels=24,
    )


def apply_cli_overrides(config: AppConfig, *, microns_per_pixel: float | None = None) -> AppConfig:
    biomass = config.biomass if microns_per_pixel is None else replace(config.biomass, microns_per_pixel=microns_per_pixel)
    return AppConfig(
        discovery=config.discovery,
        prefer_blue=config.prefer_blue,
        segmentation=config.segmentation,
        biomass=biomass,
    )


def default_config(data_roots: list[str | Path]) -> AppConfig:
    roots = [_resolve_data_root(p) for p in data_roots]
    return AppConfig(discovery=DiscoveryConfig(root_dirs=roots))
