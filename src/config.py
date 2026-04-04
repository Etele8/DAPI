from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


def default_config(data_roots: list[str | Path]) -> AppConfig:
    roots = [_resolve_data_root(p) for p in data_roots]
    return AppConfig(discovery=DiscoveryConfig(root_dirs=roots))
