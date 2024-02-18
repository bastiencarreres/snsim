"""This module contains usefull function for the survey and field geometry."""
import numpy as np
import geopandas as gpd
from shapely import geometry as shp_geo
from shapely import ops as shp_ops
from .constants import _SPHERE_LIMIT_


def _format_corner(corner, RA):
    # -- Replace corners that cross sphere edges
    #    
    #     0 ---- 1
    #     |      |
    #     3 ---- 2
    # 
    #   conditions : 
    #       - RA_0 < RA_1
    #       - RA_3 < RA_2
    #       - RA_0 and RA_3 on the same side of the field center
    # corner[fields, corner, subfields, ra/dec]
    
    sign = (corner[:, 3, :, 0] - RA[:, None]) * (corner[:, 0, :, 0] - RA[:, None]) < 0
    comp = corner[:, 0, :, 0] < corner[:, 3, :, 0]

    corner[:, 1, :, 0][corner[:, 1, :, 0] < corner[:, 0, :, 0]] += 2 * np.pi
    corner[:, 2, :, 0][corner[:, 2, :, 0] < corner[:, 3, :, 0]] += 2 * np.pi

    corner[:, 0, :, 0][sign & comp] += 2 * np.pi
    corner[:, 1, :, 0][sign & comp] += 2 * np.pi

    corner[:, 2, :, 0][sign & ~comp] += 2 * np.pi
    corner[:, 3, :, 0][sign & ~comp] += 2 * np.pi
    return corner


def _compute_area(polygon):
    """Compute survey total area."""
    # It's an integration by dec strip
    area = 0
    strip_dec = np.linspace(-np.pi/2, np.pi/2, 10_000)
    for da, db in zip(strip_dec[:-1], strip_dec[1:]):
        line = shp_geo.LineString([[0, (da + db) * 0.5], [2 * np.pi, (da + db) * 0.5]])
        if line.intersects(polygon):
            dRA = line.intersection(polygon).length
            area += dRA * (np.sin(db) - np.sin(da))
    return area


def _compute_polygon(corners):
    """Create polygon on a sphere, check for edges conditions.
    
    Notes
    -----
    corners[corner, subfields, ra/dec]
    """
    
    # Create polygons
    polygons = gpd.GeoSeries([shp_geo.Polygon(corners[:, j, :]) for j in range(corners.shape[1])])
    
    # Check if they intersect the 2pi edge line
    int_mask = polygons.intersects(_SPHERE_LIMIT_)
    
    # If they do cut divide them in 2 and translate the one that is beyond the edge at -2pi
    polydiv = gpd.GeoSeries(shp_ops.polygonize(polygons[int_mask].boundary.union(_SPHERE_LIMIT_)))
    transl_mask = polydiv.boundary.bounds['maxx'] > 2 * np.pi
    polydiv[transl_mask] = polydiv[transl_mask].translate(-2*np.pi)
    
    return shp_geo.MultiPolygon([*polygons[~int_mask].values, *polydiv.values])
