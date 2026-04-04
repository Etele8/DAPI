from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.biomass import BiomassConfig, BiomassPipelineResult, run_biomass_stage
from src.io.loader import LoadedSample
from src.preprocessing.mask_artifacts import combine_masks, create_border_mask, create_scale_bar_mask
from src.segmentation.segment import (
    SegmentationConfig,
    SegmentationResult,
    save_debug_outputs,
    segment_cells,
)


@dataclass(slots=True)
class PipelineOutput:
    sample_stem: str
    artifact_mask: np.ndarray
    segmentation: SegmentationResult
    debug_files: dict[str, Path]
    biomass: BiomassPipelineResult | None = None


def build_artifact_mask(image_gray_u8: np.ndarray) -> np.ndarray:
    return combine_masks(
        create_border_mask(image_gray_u8, margin=1),
        create_scale_bar_mask(image_gray_u8),
    )


def run_segmentation_for_sample(
    sample: LoadedSample,
    output_dir: str | Path,
    config: SegmentationConfig | None = None,
    artifact_mask: np.ndarray | None = None,
    biomass_config: BiomassConfig | None = None,
) -> PipelineOutput:
    if sample.image_bgr_u8 is None:
        raise ValueError(
            f"Sample '{sample.stem}' does not have a blue/BGR image; the rebuilt segmentation stage requires color data."
        )

    effective_artifact_mask = artifact_mask
    if effective_artifact_mask is None:
        effective_artifact_mask = build_artifact_mask(sample.image_gray_u8)

    result = segment_cells(sample.image_bgr_u8, config=config, artifact_mask=effective_artifact_mask)
    debug_dir = Path(output_dir) / f"{sample.stem}_seg"
    debug_files = save_debug_outputs(
        debug_dir,
        original_image_bgr=sample.image_bgr_u8,
        result=result,
        artifact_mask=effective_artifact_mask,
    )
    biomass = None
    if biomass_config is not None:
        biomass_dir = Path(output_dir) / f"{sample.stem}_biomass"
        biomass = run_biomass_stage(
            image_id=sample.stem,
            original_image=sample.image_bgr_u8,
            binary_mask=result.filtered_mask,
            output_dir=biomass_dir,
            config=biomass_config,
        )

    return PipelineOutput(
        sample_stem=sample.stem,
        artifact_mask=effective_artifact_mask,
        segmentation=result,
        debug_files=debug_files,
        biomass=biomass,
    )
