"""
Utility functions for Gaussian Splat generation from COLMAP output.
This module provides helpers to prepare data for 3D Gaussian Splatting training.
"""

import os
import subprocess
from pathlib import Path

def check_gaussian_splatting_available():
    """Check if 3D Gaussian Splatting training tools are available"""
    # Check for common Gaussian Splatting implementations
    try:
        # Check for 3D-GS
        result = subprocess.run(['python', '-c', 'import gsplat'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            return 'gsplat'
    except:
        pass
    
    try:
        # Check for diff-gaussian-rasterization
        result = subprocess.run(['python', '-c', 'import diff_gaussian_rasterization'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            return 'diff_gaussian'
    except:
        pass
    
    return None

def prepare_for_gaussian_splatting(colmap_output_path, images_path, output_path):
    """
    Prepare COLMAP output for Gaussian Splatting training.
    
    Args:
        colmap_output_path: Path to COLMAP sparse reconstruction (sparse/0)
        images_path: Path to input images
        output_path: Path where prepared data should be saved
    
    Returns:
        dict with paths to prepared data
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Expected structure for Gaussian Splatting:
    # - images/ (input images)
    # - sparse/0/ (COLMAP sparse reconstruction)
    
    prepared_data = {
        'images_path': None,
        'sparse_path': None,
        'ready': False
    }
    
    # Copy images if needed
    output_images = os.path.join(output_path, 'images')
    if not os.path.exists(output_images):
        os.makedirs(output_images)
        # Copy images
        import shutil
        for img_file in os.listdir(images_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(
                    os.path.join(images_path, img_file),
                    os.path.join(output_images, img_file)
                )
    prepared_data['images_path'] = output_images
    
    # Copy sparse reconstruction
    output_sparse = os.path.join(output_path, 'sparse', '0')
    os.makedirs(output_sparse, exist_ok=True)
    
    if os.path.exists(colmap_output_path):
        import shutil
        for file in ['cameras.bin', 'images.bin', 'points3D.bin']:
            src = os.path.join(colmap_output_path, file)
            dst = os.path.join(output_sparse, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        prepared_data['sparse_path'] = output_sparse
        prepared_data['ready'] = True
    
    return prepared_data

def generate_ply_from_colmap(colmap_path, output_ply_path, center_at_origin=True):
    """
    Attempt to generate a basic .ply file from COLMAP points3D.
    This creates a simple point cloud, not a full Gaussian splat.
    For full Gaussian splats, training is required.
    
    Args:
        colmap_path: Path to COLMAP reconstruction
        output_ply_path: Output PLY file path
        center_at_origin: If True, center the point cloud at origin
    """
    try:
        # Try to use pycolmap to read COLMAP data
        try:
            import pycolmap
            import numpy as np
            reconstruction = pycolmap.Reconstruction(colmap_path)
            
            # Extract points
            points = []
            colors = []
            
            for point3D_id, point3D in reconstruction.points3D.items():
                # pycolmap uses .xyz as a numpy array
                xyz = point3D.xyz
                points.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                # Colors are 0-255 in pycolmap
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
            
            # Write PLY file
            write_ply_file(output_ply_path, points, colors)
            print(f"Generated PLY file with {len(points)} points")
            return True
            
        except ImportError:
            # pycolmap not available
            print("pycolmap not available. Install with: pip install pycolmap")
            return False
        except Exception as e:
            print(f"Error reading COLMAP data: {e}")
            return False
            
    except Exception as e:
        print(f"Error generating PLY: {e}")
        return False

def write_ply_file(output_path, points, colors=None):
    """Write a simple PLY file from points and optional colors"""
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices
        for i, point in enumerate(points):
            if colors and i < len(colors):
                color = colors[i]
                # Colors should already be 0-255, but handle both cases
                r = int(color[0]) if color[0] > 1 else int(color[0] * 255)
                g = int(color[1]) if color[1] > 1 else int(color[1] * 255)
                b = int(color[2]) if color[2] > 1 else int(color[2] * 255)
                f.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
            else:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

def get_training_command(data_path, output_path):
    """
    Get the command to train Gaussian Splats.
    This is a template - actual implementation depends on the training library used.
    """
    # Example for 3D-GS style training
    # This would need to be adapted based on actual training library
    return f"""
    To train Gaussian Splats, you can use:
    
    1. 3D Gaussian Splatting (original):
       python train.py -s {data_path} -m {output_path}
    
    2. Or use Brush's training capabilities if available
    
    Note: Full training requires GPU and can take significant time.
    """

