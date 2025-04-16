# potato_shapes.py
import os
from svg_to_polygons import svg_to_polygon

def load_all_potatoes(folder="svg_01"):
    # List all SVG files in the folder and sort them (by filename)
    svg_files = sorted([f for f in os.listdir(folder) if f.endswith(".svg")])
    
    potato_polygons = []
    for i, f in enumerate(svg_files):
        filepath = os.path.join(folder, f)
        poly = svg_to_polygon(filepath, num_points=20)
        print(f"Loaded {f} with {len(poly)} points")
        print(f"Potato_{i+1:02} first 3 points: {poly[:3]}")
        potato_polygons.append(poly)

    return potato_polygons

if __name__ == "__main__":
    # Run this module to test loading and check differences visually.
    polys = load_all_potatoes(folder="svg_01")
    for idx, poly in enumerate(polys, start=1):
        print(f"Potato_{idx:02} bounding box: {min(poly, key=lambda p: p[0])} to {max(poly, key=lambda p: p[0])}")
