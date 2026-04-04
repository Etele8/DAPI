from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Calibration:
    microns_per_pixel: float | None = None

    def __post_init__(self) -> None:
        if self.microns_per_pixel is not None and self.microns_per_pixel <= 0:
            raise ValueError("microns_per_pixel must be > 0 when provided")

    @property
    def is_available(self) -> bool:
        return self.microns_per_pixel is not None

    def length_um(self, length_px: float) -> float:
        if self.microns_per_pixel is None:
            return float("nan")
        return float(length_px * self.microns_per_pixel)

    def area_um2(self, area_px2: float) -> float:
        if self.microns_per_pixel is None:
            return float("nan")
        scale = self.microns_per_pixel**2
        return float(area_px2 * scale)

    def volume_um3(self, volume_px3: float) -> float:
        if self.microns_per_pixel is None:
            return float("nan")
        scale = self.microns_per_pixel**3
        return float(volume_px3 * scale)
