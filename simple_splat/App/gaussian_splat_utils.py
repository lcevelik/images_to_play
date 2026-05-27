"""
Utility functions for Gaussian Splat generation from COLMAP output.
This module provides helpers to prepare data for 3D Gaussian Splatting training.
"""

import os
from pathlib import Path

def generate_ply_from_colmap(colmap_path, output_ply_path, center_at_origin=True):
    """
    Generate a basic .ply file from COLMAP points3D.
    This creates a simple point cloud, not a full Gaussian splat.
    For full Gaussian splats, training is required.

    Args:
        colmap_path: Path to COLMAP reconstruction
        output_ply_path: Output PLY file path
        center_at_origin: If True, center the point cloud at origin
    """
    try:
        import pycolmap
        import numpy as np
        reconstruction = pycolmap.Reconstruction(colmap_path)

        # Extract points
        points = []
        colors = []

        for point3D_id, point3D in reconstruction.points3D.items():
            xyz = point3D.xyz
            points.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
            color = point3D.color
            colors.append([int(color[0]), int(color[1]), int(color[2])])

        if len(points) == 0:
            print("No points found in reconstruction")
            return False

        # Center the point cloud at origin
        if center_at_origin:
            points_array = np.array(points)
            centroid = np.mean(points_array, axis=0)
            points_array = points_array - centroid
            points = points_array.tolist()
            print(f"Centered point cloud. Original centroid: {centroid}")

        # Write PLY file (binary for speed)
        write_ply_file(output_ply_path, points, colors)
        print(f"Generated PLY file with {len(points)} points")
        return True

    except ImportError:
        print("pycolmap not available. Install with: pip install pycolmap")
        return False
    except Exception as e:
        print(f"Error generating PLY: {e}")
        return False

def write_ply_file(output_path, points, colors=None):
    """Write a binary PLY file from points and optional colors"""
    import struct
    with open(output_path, 'wb') as f:
        # PLY header (ASCII)
        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += f"element vertex {len(points)}\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        if colors:
            header += "property uchar red\n"
            header += "property uchar green\n"
            header += "property uchar blue\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))

        # Write vertices (binary)
        for i, point in enumerate(points):
            if colors and i < len(colors):
                color = colors[i]
                r = int(color[0])
                g = int(color[1])
                b = int(color[2])
                f.write(struct.pack('<fffBBB', point[0], point[1], point[2], r, g, b))
            else:
                f.write(struct.pack('<fff', point[0], point[1], point[2]))
