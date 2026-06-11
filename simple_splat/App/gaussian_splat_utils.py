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

def prune_gaussian_ply(input_ply_path, output_ply_path=None, opacity_threshold=0.005):
    """Remove near-invisible Gaussians from a 3DGS PLY file.

    Keeps only Gaussians where sigmoid(raw_opacity) >= opacity_threshold.
    Typically removes 40-60% of Gaussians with no perceptible quality change,
    matching LichtFeld-Studio's default prune-ratio=0.6 behaviour.

    Args:
        input_ply_path: Path to input 3DGS PLY (binary little-endian)
        output_ply_path: Output path — defaults to overwriting the input
        opacity_threshold: Min opacity to keep (0.005 = half a percent)

    Returns:
        (original_count, kept_count) or (0, 0) on failure
    """
    import numpy as np

    if output_ply_path is None:
        output_ply_path = input_ply_path

    try:
        with open(input_ply_path, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='replace').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            data_bytes = f.read()

        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append((parts[2], parts[1]))

        type_map = {'float': np.float32, 'double': np.float64,
                    'uchar': np.uint8, 'int': np.int32, 'uint': np.uint32}
        dtype = np.dtype([(n, type_map.get(t, np.float32)) for n, t in properties])

        data = np.frombuffer(data_bytes, dtype=dtype, count=num_vertices)

        prop_names = [p[0] for p in properties]
        if 'opacity' not in prop_names:
            return 0, 0  # Not a 3DGS PLY

        raw_opacity = data['opacity'].astype(np.float64)
        actual_opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
        mask = actual_opacity >= opacity_threshold
        pruned = data[mask]

        new_header = '\n'.join(
            f'element vertex {len(pruned)}' if l.startswith('element vertex') else l
            for l in header_lines
        ) + '\n'

        with open(output_ply_path, 'wb') as f:
            f.write(new_header.encode('ascii'))
            f.write(pruned.tobytes())

        return num_vertices, len(pruned)

    except Exception as e:
        print(f"PLY pruning failed: {e}")
        return 0, 0


def write_ply_file(output_path, points, colors=None):
    """Write a binary PLY file from points and optional colors.
    Uses batched numpy tobytes() for ~10x faster writing vs per-vertex struct.pack."""
    import numpy as np
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

        # Batch write using numpy tobytes() — ~10x faster than per-vertex struct.pack
        pts = np.array(points, dtype=np.float32)
        if colors:
            cols = np.array(colors, dtype=np.uint8)
            # Interleave: for each vertex, write 3 floats + 3 uchars
            # Build a structured array
            vertex_dtype = np.dtype([('xyz', np.float32, 3), ('rgb', np.uint8, 3)])
            vertices = np.empty(len(pts), dtype=vertex_dtype)
            vertices['xyz'] = pts
            vertices['rgb'] = cols
            f.write(vertices.tobytes())
        else:
            f.write(pts.tobytes())
