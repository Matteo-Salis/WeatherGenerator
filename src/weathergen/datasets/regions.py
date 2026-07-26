# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Each entry maps a region name to a geographic bounding box
(lat_min, lat_max, lon_min, lon_max) in degrees.
Add a new region by inserting one line here
"""

NAMED_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global": (-90.0, 90.0, -180.0, 180.0),
    "nhem": (0.0, 90.0, -180.0, 180.0),
    "shem": (-90.0, 0.0, -180.0, 180.0),
    "tropics": (-30.0, 30.0, -180.0, 180.0),
    "belgium": (49.0, 52.0, 2.0, 7.0),
    "europe": (35.0, 70.0, -10.0, 40.0),
    "cerra": (18.0, 77.0, -60.0, 75.0),
    "arctic": (50.0, 90.0, -180.0, 180.0),
    "uwc-west": (39.0, 63.0, -26.0, 41.0),
    "arome": (37.0, 56.0, -12.0, 16.0),
    "icon": (42.0, 51.0, -1.0, 18.0),
    "ch": (45.5, 48.0, 5.5, 11.0),
}


def get_region_box(name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``((lat_min, lat_max), (lon_min, lon_max))`` for a named region"""
    key = str(name).lower()
    assert key in NAMED_REGIONS, f"Unknown region name '{name}'. Known: {sorted(NAMED_REGIONS)}."
    lat_min, lat_max, lon_min, lon_max = NAMED_REGIONS[key]
    return (lat_min, lat_max), (lon_min, lon_max)
