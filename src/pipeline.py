from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.annotation_manifest import write_annotation_manifest
from src.biomass import BiomassPipelineResult, run_biomass_stage
from src.config import AppConfig, BiomassConfig, SegmentationConfig
from src.crop_export import CropExportResult, export_candidate_crops
from src.io.loader import LoadedSample
from src.local_refinement import RefinedCropRecord, refine_positive_crops
from src.preprocessing.mask_artifacts import combine_masks, create_border_mask, create_scale_bar_mask
from src.proposal_generation import (
    ProposalGenerationResult,
    generate_proposals,
    save_proposal_debug_outputs,
    write_candidate_records,
)
from src.segmentation.segment import (
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


@dataclass(slots=True)
class ProposalPipelineOutput:
    sample_stem: str
    artifact_mask: np.ndarray
    proposals: ProposalGenerationResult
    debug_files: dict[str, Path]
    candidate_files: dict[str, Path]
    crop_export: CropExportResult | None
    annotation_files: dict[str, Path]


@dataclass(slots=True)
class RefinementPipelineOutput:
    refined_records: list[RefinedCropRecord]
    files: dict[str, Path]


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


def run_proposal_pipeline_for_sample(
    sample: LoadedSample,
    output_dir: str | Path,
    *,
    app_config: AppConfig,
    artifact_mask: np.ndarray | None = None,
) -> ProposalPipelineOutput:
    if sample.image_bgr_u8 is None:
        raise ValueError(
            f"Sample '{sample.stem}' does not have a blue/BGR image; proposal generation requires color data."
        )

    effective_artifact_mask = artifact_mask
    if effective_artifact_mask is None:
        effective_artifact_mask = build_artifact_mask(sample.image_gray_u8)

    sample_output_dir = Path(output_dir) / sample.stem
    proposals = generate_proposals(
        image_id=sample.stem,
        image_bgr=sample.image_bgr_u8,
        segmentation_config=app_config.segmentation,
        filter_config=app_config.proposal.filter,
        profile_name=app_config.profile_name,
        artifact_mask=effective_artifact_mask,
    )
    debug_files = save_proposal_debug_outputs(
        sample_output_dir / app_config.proposal.debug_subdir_name,
        original_image_bgr=sample.image_bgr_u8,
        proposal_result=proposals,
    )
    candidate_files = write_candidate_records(
        sample_output_dir,
        proposals.candidates,
        csv_name=app_config.proposal.candidates_csv_name,
        jsonl_name=app_config.proposal.candidates_jsonl_name,
    )

    crop_export = None
    annotation_files: dict[str, Path] = {}
    if app_config.proposal.crop_export.enabled:
        crop_export = export_candidate_crops(
            image_bgr=sample.image_bgr_u8,
            proposal_result=proposals,
            output_dir=sample_output_dir,
            config=app_config.proposal.crop_export,
        )
        annotation_files = write_annotation_manifest(
            crop_export.records,
            output_dir=sample_output_dir,
            config=app_config.proposal.annotation,
        )

    return ProposalPipelineOutput(
        sample_stem=sample.stem,
        artifact_mask=effective_artifact_mask,
        proposals=proposals,
        debug_files=debug_files,
        candidate_files=candidate_files,
        crop_export=crop_export,
        annotation_files=annotation_files,
    )


def run_refinement_from_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    app_config: AppConfig,
    run_biomass: bool = False,
) -> RefinementPipelineOutput:
    refined_records, files = refine_positive_crops(
        manifest_path,
        output_dir=output_dir,
        refinement_config=app_config.proposal.refinement,
        dataset_config=app_config.proposal.dataset,
        biomass_config=app_config.biomass,
        run_biomass=run_biomass,
    )
    return RefinementPipelineOutput(refined_records=refined_records, files=files)
