# DAPI Proposal-to-Biomass Pipeline

This repository implements a microscopy workflow for DAPI-like fluorescence images where the first segmentation pass is used to generate candidate objects, not final scientific measurements.

The core strategy is:

1. Run high-recall full-image segmentation to propose possible cells.
2. Export crops around those proposals for human annotation.
3. Prepare a classifier dataset from the annotated crops.
4. Refine contours locally on accepted positive crops.
5. Compute biomass and biovolume from refined contours rather than from the loose proposal mask.

This design shifts effort away from forcing perfect full-image segmentation too early and toward a more robust proposal, review, refinement, and measurement pipeline.

## Current Status

Implemented:

- bundled TOML profiles, including a dedicated `proposal_high_recall` profile
- full-image segmentation and biomass pipeline
- proposal generation with per-candidate records
- crop export with stable filenames and metadata
- annotation manifest export
- classifier dataset preparation hooks with image-level split grouping
- local refinement on accepted crops
- optional biomass computation on refined masks
- CLI entry points for all implemented stages

Not yet implemented:

- model training and inference for the crop classifier
- direct ingestion of classifier prediction files into refinement
- automated end-to-end proposal -> classifier -> refinement orchestration in one command

## Repository Layout

Key files:

- `src/cli.py`: command-line entry point
- `src/config.py`: typed config model and TOML profile loading
- `src/profiles/default.toml`: default segmentation profile
- `src/profiles/proposal_high_recall.toml`: high-recall proposal profile
- `src/proposal_generation.py`: proposal extraction and proposal debug outputs
- `src/crop_export.py`: crop export and crop metadata generation
- `src/annotation_manifest.py`: annotation-ready CSV manifest writer
- `src/classifier_dataset.py`: manifest ingestion and train/val/test split generation
- `src/local_refinement.py`: local crop refinement and optional refined-crop biomass
- `src/pipeline.py`: orchestration helpers for segmentation and proposal workflows
- `src/biomass/volume_pipeline.py`: contour-based biomass and biovolume measurement

## Installation

Python 3.11 or newer is required.

Install in editable mode:

```bash
pip install -e .
```

Available runtime dependencies are intentionally minimal:

- `numpy`
- `opencv-python`

## Profiles

Profiles are loaded from `src/profiles/*.toml`.

Bundled profiles:

- `default`: balanced segmentation settings
- `calibrated_0645`: includes biomass calibration metadata
- `proposal_high_recall`: tuned for candidate generation, crop export, and later refinement

List available profiles:

```bash
python -m src.cli --list-profiles
```

## CLI Usage

The main script is:

```bash
python -m src.cli
```

Global options:

- `--data-root`: root directory to scan for samples, repeatable
- `--sample`: process a single discovered sample
- `--output-dir`: output root directory
- `--profile`: bundled profile name
- `--profile-file`: custom TOML profile path
- `--microns-per-pixel`: calibration override for biomass conversion

### 1. Full-Image Segmentation

Run the original segmentation and biomass flow:

```bash
python -m src.cli --mode segment --profile default --data-root data/samples --sample Tv143 --output-dir outputs/segment
```

### 2. Proposal Generation

Run high-recall proposal generation and crop export:

```bash
python -m src.cli --mode proposals --profile proposal_high_recall --data-root data/samples --sample Tv143 --output-dir outputs/proposals
```

This writes per-sample outputs such as:

- `proposal_candidates.csv`
- `proposal_candidates.jsonl`
- `proposal_debug/`
- `crops/images/`
- `crops/overlays/`
- `crops/masks/`
- `annotation/annotation_manifest.csv`

### 3. Prepare Classifier Dataset

After filling `label` in the exported annotation manifest:

```bash
python -m src.cli --mode prepare-dataset --profile proposal_high_recall --manifest outputs/proposals/Tv143/annotation/annotation_manifest.csv --output-dir outputs/classifier
```

This produces:

- `classifier_dataset/train.csv`
- `classifier_dataset/val.csv`
- `classifier_dataset/test.csv`
- `classifier_dataset/split_summary.json`

Splits are grouped by `image_id` by default to reduce leakage between train and validation.

### 4. Local Refinement

Run refinement on positively labeled crops:

```bash
python -m src.cli --mode refine --profile proposal_high_recall --manifest outputs/proposals/Tv143/annotation/annotation_manifest.csv --output-dir outputs/refined
```

Run refinement and refined-crop biomass:

```bash
python -m src.cli --mode refine --profile proposal_high_recall --manifest outputs/proposals/Tv143/annotation/annotation_manifest.csv --output-dir outputs/refined --run-biomass
```

## Expected Annotation Labels

The manifest is intentionally not restricted to binary labeling. The default proposal profile is prepared for labels such as:

- `single_valid`
- `invalid`
- `merged`
- `uncertain`

Classifier dataset preparation currently maps labels using the proposal profile config:

- positive labels: `single_valid`
- negative labels: `invalid`, `merged`
- excluded labels: `uncertain`

## Data Discovery

The loader discovers samples from one or more roots and groups common microscopy naming variants into logical sample stems.

Supported inputs include:

- BGR or blue-channel images
- grayscale variants
- annotated companion images
- report text files
- report spreadsheets

The current loading path prefers color or blue-source images when available because the rebuilt segmentation stage uses color evidence.

## Outputs And Debugging

The repository is designed to be auditable at each stage.

Full-image segmentation outputs include:

- original image
- evidence maps
- raw and cleaned masks
- final filtered mask
- region overlays

Proposal outputs include:

- proposal mask
- connected-component labels
- candidate overlays
- candidate IDs on image
- crop-box overview

Crop outputs include:

- crop image
- crop overlay
- crop mask preview
- crop metadata CSV and JSONL

Refinement outputs include:

- refined masks
- refined overlays
- refined crop metadata
- optional refined-crop biomass outputs

## Scientific Intent

The pipeline is built around the idea that biomass matters more than raw counts and that contour quality matters more than early-stage precision when estimating morphology-dependent volume.

The downstream target remains contour-based biovolume estimation in the Zeder or YABBA spirit:

- first-pass segmentation is for recall and candidate discovery
- human review or classifier filtering removes false positives
- local refinement improves contour fidelity
- biomass is computed only from refined contours

## Testing

Run the unit tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Current tests cover:

- baseline segmentation on synthetic images
- debug output writing
- proposal export and annotation manifest creation
- image-level classifier split grouping
- positive-crop local refinement

## Recommended Next Steps

The next high-value additions are:

1. a real crop classifier training module
2. prediction export keyed by `crop_id`
3. refinement that can consume predictions instead of manual labels
4. aggregate biomass summaries across refined crops and source images

## License

No license file is currently included in the repository. Add one before distributing the project publicly.
