# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pathlib
import warnings
from collections.abc import Sequence

import anemoi.datasets as anemoi_datasets
import astropy_healpix as hp
import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon
from weathergen.datasets.regions import get_region_box
from weathergen.datasets.utils import coords_to_hpyidxs


def geographic_cell_centers(level: int) -> tuple[NDArray, NDArray]:
    """Cell-centre (lat_deg, lon_deg) in the tokenizer's coordinate convention."""
    num = 12 * 4**level
    lon, lat = hp.healpix_to_lonlat(np.arange(num), 2**level, order="nested")
    geo_lon = lon.deg - 180.0
    return lat.deg.astype(np.float64), geo_lon.astype(np.float64)


def _buffer_by_rings(selected: NDArray, level: int, rings: int) -> NDArray:
    """Return ``selected`` grown by ``rings`` layers of neighbouring HEALPix cells."""
    if rings <= 0:
        return selected

    in_buffer = np.zeros(12 * 4**level, dtype=np.bool_)
    in_buffer[selected] = True

    for _ in range(rings):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            temp = hp.neighbours(np.flatnonzero(in_buffer), 2**level, order="nested").transpose()
        valid = temp != -1
        in_buffer[temp[valid]] = True

    return np.sort(np.flatnonzero(in_buffer))


def _box_cells(level: int, region: dict) -> NDArray:
    """Cells whose centre lies inside the region's geographic box."""
    name = region.get("name")
    if name is not None:
        lat_rng, lon_rng = get_region_box(name)
    else:
        lat_rng, lon_rng = region.get("lat"), region.get("lon")
    valid = lat_rng is not None and lon_rng is not None and len(lat_rng) == 2 and len(lon_rng) == 2
    assert valid, (
        "healpix_active_region requires a known 'name', explicit lat: [min, max] and "
        "lon: [min, max], or from_data: true."
    )
    lat_min, lat_max = float(lat_rng[0]), float(lat_rng[1])
    lon_min, lon_max = float(lon_rng[0]), float(lon_rng[1])

    lat, lon = geographic_cell_centers(level)
    if lon_min <= lon_max:
        lon_in = (lon >= lon_min) & (lon <= lon_max)
    else:  # box wraps the antimeridian
        lon_in = (lon >= lon_min) | (lon <= lon_max)

    return np.flatnonzero(lon_in & (lat >= lat_min) & (lat <= lat_max)).astype(np.int64)


def _coverage_cells(level: int, stream_info: dict, data_paths: Sequence) -> NDArray:
    """Cells holding at least one data point of the stream 
    For projected grids this avoids the empty cells a lat/lon bounding box inevitably includes.
    """
    assert stream_info.get("type") == "anemoi", (
        "healpix_active_region 'from_data' is only supported for anemoi streams; stream "
        f"'{stream_info.get('name')}' has type '{stream_info.get('type')}'."
    )
    filenames = stream_info.get("filenames", [])
    assert filenames, "healpix_active_region 'from_data' requires the stream to set 'filenames'"

    crop = {k: stream_info[k] for k in ("area", "trim_edge", "thinning") if k in stream_info}
    crop = {
        k: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
        for k, v in crop.items()
    }

    cells = []
    for filename in filenames:
        # streams name their datasets, so look them up under data_paths as the sampler does
        candidates = [pathlib.Path(filename), *(pathlib.Path(p) / filename for p in data_paths)]
        path = next((p for p in candidates if p.exists()), None)
        assert path is not None, f"healpix_active_region 'from_data': not found: {candidates}"

        ds = anemoi_datasets.open_dataset(str(path), **crop)
        # clip exactly as the reader does, so cells match the tokenizer's point assignment
        lats = _clip_lat(ds.latitudes).astype(np.float64)
        lons = _clip_lon(ds.longitudes).astype(np.float64)
        cells.append(coords_to_hpyidxs(level, lats, lons))

    return np.unique(np.concatenate(cells)).astype(np.int64)


def resolve_region_cells(
    level: int, region: dict, stream_info: dict | None = None, data_paths: Sequence = ()
) -> NDArray:
    """Native nested cell ids of a region, grown by a ``rings`` buffer of neighbouring cells.

    The region is either a geographic box (``name``, or explicit ``lat``/``lon``) or, with
    ``from_data: true``, the exact footprint of a dataset. The buffer is measured
    in cells, so a fixed ring count covers a smaller geographic extent at finer levels.
    """
    if region.get("from_data", False):
        # a region naming its own dataset borrows that footprint; otherwise use the stream's
        source_info = region if region.get("filenames") else stream_info
        assert source_info is not None, (
            "healpix_active_region 'from_data' needs a dataset to resolve against: use it in a "
            "stream config, give the region its own 'filenames', or use "
            "'from_stream: <STREAM>' for fe_healpix_active_region."
        )
        selected = _coverage_cells(level, source_info, data_paths)
    else:
        selected = _box_cells(level, region)

    assert selected.size > 0, "healpix_active_region selected no healpix cells"

    return _buffer_by_rings(selected, level, int(region.get("rings", 0)))


def _stream_region(stream_info) -> dict | None:
    """Region spec of one stream: its own ``healpix_active_region``, else None (global)."""
    region = stream_info.get("healpix_active_region", None)
    if OmegaConf.is_config(region):
        region = OmegaConf.to_container(region, resolve=True)
    return region


def region_for_level(cf, level: int):
    """Active region for the encoder operating at HEALPix ``level``.
    """
    default_level = int(cf.healpix_level)
    streams = [
        si for si in cf.streams.values() if int(si.get("healpix_level", default_level)) == level
    ]
    regions = [_stream_region(si) for si in streams]

    # no stream at this level, or at least one of them covers the globe
    if not regions or any(not r for r in regions):
        return None

    # one shared box spec: pass it on unresolved, so it also reads well in the startup log
    first = regions[0]
    if not first.get("from_data", False) and all(r == first for r in regions):
        return first

    data_paths = list(cf.get("data_paths", []))
    resolved = [
        resolve_region_cells(level, r, si, data_paths)
        for si, r in zip(streams, regions, strict=True)
    ]
    return np.unique(np.concatenate(resolved))


def forecast_level(cf) -> int:
    """Level of the forecast/decode grid (grid_F)"""
    level = cf.get("fe_healpix_level", None)
    assert level is not None, "config must set 'fe_healpix_level' (the forecast/decode grid level)"
    return int(level)


def forecast_region(cf):
    """Active region for the forecast/decode grid (grid_F).

    Supports ``from_stream: <STREAM>`` to use the exact data footprint of the named stream,
    plus an optional ``rings`` buffer, as the forecast grid. A region carrying its own
    ``from_data`` + ``filenames`` is resolved here too, since NativeGrid has no data_paths
    with which to locate the dataset.
    """
    region = cf.get("fe_healpix_active_region", cf.get("healpix_active_region", None))
    if OmegaConf.is_config(region):
        region = OmegaConf.to_container(region, resolve=True)

    if isinstance(region, dict) and region.get("from_data", False) and region.get("filenames"):
        return resolve_region_cells(
            forecast_level(cf), region, None, list(cf.get("data_paths", []))
        )

    name = region.get("from_stream") if isinstance(region, dict) else None
    if name is None:
        return region

    assert name in cf.streams, (
        f"fe_healpix_active_region from_stream '{name}' is not a configured stream: "
        f"{list(cf.streams.keys())}"
    )
    return resolve_region_cells(
        forecast_level(cf),
        {"from_data": True, "rings": region.get("rings", 0)},
        cf.streams[name],
        list(cf.get("data_paths", [])),
    )


def region_label(region) -> str:
    """Compact description of a region spec, for logging."""
    if OmegaConf.is_config(region):
        region = OmegaConf.to_container(region, resolve=True)
    if region is None or len(region) == 0:
        return "global"
    if isinstance(region, dict):
        return str(region)
    return f"{len(region)} explicit cells"


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

    def __init__(self, cf, level: int | None = None, region=None) -> None:
        self.level = int(cf.healpix_level if level is None else level)
        self.num_global = 12 * 4**self.level

        if OmegaConf.is_config(region):
            region = OmegaConf.to_container(region, resolve=True)

        if region is None or len(region) == 0:
            self.active_to_global = np.arange(self.num_global, dtype=np.int64)
        elif isinstance(region, dict):
            self.active_to_global = resolve_region_cells(self.level, region)
        else:  # explicit cell ids, e.g. a data-driven or unioned region
            self.active_to_global = np.unique(np.asarray(region, dtype=np.int64))

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
