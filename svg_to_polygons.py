# svg_to_polygon.py
from svgpathtools import svg2paths
import numpy as np

def svg_to_polygon(filepath, num_points=20):
    # Load all paths from the SVG file.
    paths, _ = svg2paths(filepath)
    print(f"{filepath} has {len(paths)} paths")
    if not paths:
        raise ValueError("No paths found in the SVG file.")
    
    # Sample points from ALL paths
    points = []
    for p_idx, path in enumerate(paths):
        for i in range(num_points):
            t = i / num_points
            point = path.point(t)
            points.append((point.real, point.imag))
    
    return normalize_polygon(points)

def normalize_polygon(points):
    points = np.array(points)

    # Center around origin
    centroid = points.mean(axis=0)
    points -= centroid

    # Scale to max ~100 units
    width = np.ptp(points[:, 0])
    height = np.ptp(points[:, 1])
    max_dim = max(width, height)
    scale = 100 / max_dim if max_dim != 0 else 1.0
    points *= scale

    return [tuple(pt) for pt in points]
