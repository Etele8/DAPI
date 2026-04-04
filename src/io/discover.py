from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import DiscoveryConfig

_GRAY_SUFFIX_RE = re.compile(r"(?i)(?:[_\-\s]?(?:grayscale|greyscale|gray|grey))$")


@dataclass(slots=True)
class SampleFiles:
    stem: str
    folder: Path
    blue: Path | None = None
    gray: Path | None = None
    annotated: Path | None = None
    report_txt: Path | None = None
    report_xlsx: Path | None = None
    extras: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["folder"] = str(self.folder)
        for key in ("blue", "gray", "annotated", "report_txt", "report_xlsx"):
            if out[key] is not None:
                out[key] = str(out[key])
        out["extras"] = [str(x) for x in self.extras]
        return out


def _normalize_stem(name: str, cfg: DiscoveryConfig) -> str:
    """
    Reduce different file naming variants to one logical stem.

    Examples:
    - Tv143.png -> Tv143
    - Tv143_grayscale.tif -> Tv143
    - Tv143.tif-annotated.jpg -> Tv143
    - Tv143.tif-report.txt -> Tv143
    - Tv143-report.xlsx -> Tv143
    """
    base = name

    for suffix in (*cfg.annotated_suffixes, *cfg.report_txt_suffixes, *cfg.report_xlsx_suffixes):
        if base.lower().endswith(suffix.lower()):
            base = base[: -len(suffix)]
            break

    # remove one normal extension if present after trimming report/annotated suffix
    base_path = Path(base)
    if base_path.suffix:
        base = base_path.with_suffix("").name

    # remove gray suffix variants from the end
    base = _GRAY_SUFFIX_RE.sub("", base)

    return base


def _is_annotated_file(path: Path, cfg: DiscoveryConfig) -> bool:
    return any(path.name.lower().endswith(s.lower()) for s in cfg.annotated_suffixes)


def _is_report_txt(path: Path, cfg: DiscoveryConfig) -> bool:
    return any(path.name.lower().endswith(s.lower()) for s in cfg.report_txt_suffixes)


def _is_report_xlsx(path: Path, cfg: DiscoveryConfig) -> bool:
    return any(path.name.lower().endswith(s.lower()) for s in cfg.report_xlsx_suffixes)


def _looks_like_gray(path: Path, cfg: DiscoveryConfig) -> bool:
    lower_name = path.stem.lower()
    return bool(_GRAY_SUFFIX_RE.search(lower_name)) or any(
        marker.lower() in lower_name for marker in cfg.gray_name_markers
    )


def _is_excel(path: Path) -> bool:
    return path.suffix.lower() == ".xlsx"


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _choose_best(existing: Path | None, candidate: Path) -> Path:
    """
    Keep the shorter / cleaner filename if both qualify.
    """
    if existing is None:
        return candidate
    return candidate if len(candidate.name) < len(existing.name) else existing


def discover_samples(cfg: DiscoveryConfig) -> list[SampleFiles]:
    grouped: dict[tuple[Path, str], SampleFiles] = {}

    for root in cfg.root_dirs:
        if not root.exists():
            raise FileNotFoundError(f"Data root does not exist: {root}")

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if not (_is_image(path) or _is_excel(path) or path.suffix.lower() == ".txt"):
                continue

            stem = _normalize_stem(path.name, cfg)
            key = (path.parent, stem)

            if key not in grouped:
                grouped[key] = SampleFiles(stem=stem, folder=path.parent)

            sample = grouped[key]

            if _is_annotated_file(path, cfg):
                sample.annotated = _choose_best(sample.annotated, path)
            elif _is_report_txt(path, cfg):
                sample.report_txt = _choose_best(sample.report_txt, path)
            elif _is_report_xlsx(path, cfg):
                sample.report_xlsx = _choose_best(sample.report_xlsx, path)
            elif _is_image(path):
                if _looks_like_gray(path, cfg):
                    sample.gray = _choose_best(sample.gray, path)
                else:
                    # raw blue or maybe plain gray fallback like Tv143.tif
                    # prefer plain tif as gray only if no explicit blue exists later
                    sample.extras.append(path)
            else:
                sample.extras.append(path)

    # resolve extras into blue/gray if still missing
    for sample in grouped.values():
        image_extras = [p for p in sample.extras if _is_image(p)]

        # explicit gray already found, keep it
        # choose blue from png/jpg first, then tif as fallback
        if sample.blue is None:
            blue_candidates = [p for p in image_extras if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            tif_candidates = [p for p in image_extras if p.suffix.lower() in {".tif", ".tiff"}]

            if blue_candidates:
                sample.blue = sorted(blue_candidates, key=lambda p: len(p.name))[0]
            elif tif_candidates and sample.gray is None:
                # if only tif exists and no explicit gray marker, treat as gray
                sample.gray = sorted(tif_candidates, key=lambda p: len(p.name))[0]
            elif tif_candidates:
                sample.blue = sorted(tif_candidates, key=lambda p: len(p.name))[0]

        # if gray still missing, maybe plain tif should be gray
        if sample.gray is None:
            tif_candidates = [p for p in image_extras if p.suffix.lower() in {".tif", ".tiff"}]
            if tif_candidates:
                sample.gray = sorted(tif_candidates, key=lambda p: len(p.name))[0]

    samples = sorted(grouped.values(), key=lambda s: (str(s.folder), s.stem))
    return samples
