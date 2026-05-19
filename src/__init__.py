"""Top-level public API for the DAPI pipeline.

Config dataclasses (`SegmentationConfig`, `BiomassConfig`, etc.) live in
`src.config` and are imported from there directly. This module re-exports
only the high-level workflow entry points and their result types.
"""

from src.annotation_review import (
    AnnotationRunResult,
    AnnotationSessionResult,
    annotate_manifest,
    annotate_root,
    build_annotation_manifest_from_images,
    collect_annotation_manifests,
)
from src.biomass import (
    BiomassDebugFiles,
    BiomassObjectMeasurement,
    BiomassPipelineResult,
    BiomassSummary,
    Calibration,
    run_biomass_stage,
)
from src.classifier_dataset import ingest_annotation_manifest, prepare_classifier_dataset
from src.crop_export import CropExportResult, CropRecord, export_candidate_crops
from src.local_refinement import RefinedCropRecord, refine_positive_crops
from src.pipeline import (
    PipelineOutput,
    ProposalPipelineOutput,
    RefinementPipelineOutput,
    build_artifact_mask,
    run_proposal_pipeline_for_sample,
    run_refinement_from_manifest,
    run_segmentation_for_sample,
)
from src.proposal_generation import ProposalCandidate, ProposalGenerationResult, generate_proposals
from src.segmentation import (
    RegionDetection,
    SegmentationResult,
    save_debug_outputs,
    segment_cells,
    segment_objects,
)

__all__ = [
    "AnnotationRunResult",
    "AnnotationSessionResult",
    "BiomassDebugFiles",
    "BiomassObjectMeasurement",
    "BiomassPipelineResult",
    "BiomassSummary",
    "Calibration",
    "CropExportResult",
    "CropRecord",
    "PipelineOutput",
    "ProposalCandidate",
    "ProposalGenerationResult",
    "ProposalPipelineOutput",
    "RefinedCropRecord",
    "RefinementPipelineOutput",
    "RegionDetection",
    "SegmentationResult",
    "annotate_manifest",
    "annotate_root",
    "build_annotation_manifest_from_images",
    "build_artifact_mask",
    "collect_annotation_manifests",
    "export_candidate_crops",
    "generate_proposals",
    "ingest_annotation_manifest",
    "prepare_classifier_dataset",
    "refine_positive_crops",
    "run_biomass_stage",
    "run_proposal_pipeline_for_sample",
    "run_refinement_from_manifest",
    "run_segmentation_for_sample",
    "save_debug_outputs",
    "segment_cells",
    "segment_objects",
]
