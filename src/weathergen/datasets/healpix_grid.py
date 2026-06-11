# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

import astropy_healpix as hp
import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from weathergen.datasets.regions import get_region_box

EARTH_RADIUS_KM = 6371.0


def geographic_cell_centers(level: int) -> tuple[NDArray, NDArray]:
    """Cell-centre (lat_deg, lon_deg) in the tokenizer's coordinate convention.
    """
    num = 12 * 4**level
    lon, lat = hp.healpix_to_lonlat(np.arange(num), 2**level, order="nested")
    geo_lon = lon.deg - 180.0
    return lat.deg.astype(np.float64), geo_lon.astype(np.float64)


def _buffer_by_km(selected: NDArray, level: int, buffer_km: float) -> NDArray:
    """Return all cells within ``buffer_km`` great-circle distance of any cell in ``selected``.
    """
    if buffer_km <= 0:
        return selected

    num = 12 * 4**level
    x, y, z = hp.healpix_to_xyz(np.arange(num), 2**level, order="nested")
    xyz_all = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=1).astype(np.float32)
    xyz_sel = xyz_all[selected]

    cos_thresh = float(np.cos(buffer_km / EARTH_RADIUS_KM))

    chunk = max(1, int(50 * 1024 * 1024 / (num * xyz_all.itemsize)))
    in_buffer = np.zeros(num, dtype=bool)
    for i in range(0, len(selected), chunk):
        dots = xyz_all @ xyz_sel[i : i + chunk].T  
        in_buffer |= (dots >= cos_thresh).any(axis=1)

    return np.sort(np.flatnonzero(in_buffer))


def resolve_region_cells(level: int, region: dict) -> NDArray:
    """
    Native nested cell ids inside the configured geographic box, grown by a ``buffer_km``
    great-circle buffer. The buffer is resolution-independent: the same geographic extent is
    selected at every HEALPix level, which is required for multi-level encoder fusion.
    """
    name = region.get("name")
    if name is not None:
        lat_rng, lon_rng = get_region_box(name)
    else:
        lat_rng = region.get("lat")
        lon_rng = region.get("lon")
    valid = lat_rng is not None and lon_rng is not None and len(lat_rng) == 2 and len(lon_rng) == 2
    assert valid, (
        "healpix_active_region requires a known 'name' or explicit lat: [min, max] and "
        "lon: [min, max]."
    )
    lat_min, lat_max = float(lat_rng[0]), float(lat_rng[1])
    lon_min, lon_max = float(lon_rng[0]), float(lon_rng[1])

    lat, lon = geographic_cell_centers(level)

    if lon_min <= lon_max:
        lon_in = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_in = (lon >= lon_min) | (lon <= lon_max)
    sel = lon_in & (lat >= lat_min) & (lat <= lat_max)

    selected = np.flatnonzero(sel).astype(np.int64)
    assert selected.size > 0, "healpix_active_region selected no healpix cells"

    return _buffer_by_km(selected, level, float(region.get("buffer_km", 0)))


class NativeGrid:
    """Geometry of the native HEALPix grid the encoder operates on.

    Attributes
    ----------
    level : int
        Native HEALPix level.
    num_global : int
        Number of cells on the full globe at the native level.
    num_cells : int
        Number of *active* cells (== num_global when no region is configured).
    is_full : bool
        Whether the grid covers the whole globe (no active region).
    active_to_global : np.ndarray[int64], shape (num_cells,)
        Active index -> global nested cell id.
    global_to_active : np.ndarray[int64], shape (num_global,)
        Global nested cell id -> active index, ``-1`` for inactive cells.
    """

    def __init__(self, cf, level: int | None = None) -> None:
        self.level = int(cf.healpix_level if level is None else level)
        self.num_global = 12 * 4**self.level

        region = cf.get("healpix_active_region", None)
        if region is not None and OmegaConf.is_config(region):
            region = OmegaConf.to_container(region, resolve=True)

        if not region:
            self.active_to_global = np.arange(self.num_global, dtype=np.int64)
        else:
            self.active_to_global = resolve_region_cells(self.level, region)

        self.num_cells = int(self.active_to_global.shape[0])
        self.is_full = self.num_cells == self.num_global

        g2a = np.full(self.num_global, -1, dtype=np.int64)
        g2a[self.active_to_global] = np.arange(self.num_cells, dtype=np.int64)
        self.global_to_active = g2a

    @property
    def nside(self) -> int:
        return 2**self.level

    def active_to_global_tensor(self, device=None) -> torch.Tensor:
        return torch.from_numpy(self.active_to_global).to(device=device, dtype=torch.long)

    def global_to_active_tensor(self, device=None) -> torch.Tensor:
        return torch.from_numpy(self.global_to_active).to(device=device, dtype=torch.long)

    def neighbours_self_filled(self) -> torch.Tensor:
        """
        (num_cells, 9) self + 8 neighbours in *active* index space,
        with missing (pole) or out-of-region neighbours filled with the cell itself
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            temp = hp.neighbours(self.active_to_global, self.nside, order="nested").transpose()
        # map global neighbour ids -> active ids (-1 = missing or inactive)
        temp_active = np.full_like(temp, -1)
        valid = temp != -1
        temp_active[valid] = self.global_to_active[temp[valid]]

        out = np.empty((self.num_cells, temp.shape[1] + 1), dtype=np.int64)
        out[:, 0] = np.arange(self.num_cells, dtype=np.int64)
        out[:, 1:] = temp_active
        # self-fill missing / out-of-region neighbours
        for i, row in enumerate(out[:, 1:]):
            row[row == -1] = i

        return torch.from_numpy(out).to(torch.int32)
