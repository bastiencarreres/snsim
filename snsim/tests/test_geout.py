import numpy as np
import shapely.geometry as shp_geo
from numpy.testing import assert_almost_equal
from snsim import geo_utils as geo_ut


def test_compute_area():
    corners = np.array(
        [[[0, np.pi / 2]], [[2 * np.pi, np.pi / 2]], [[2 * np.pi, -np.pi / 2]], [[0, -np.pi / 2]]]
    )

    polygon = geo_ut._compute_polygon(corners)

    area = geo_ut._compute_area(polygon)
    assert_almost_equal(area, 4 * np.pi)
