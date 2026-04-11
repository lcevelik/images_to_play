"""
Dense reconstruction module for COLMAP
This module performs dense stereo matching and depth map fusion to generate
high-density point clouds (millions of points vs. thousands from sparse SfM)
"""

import os
import subprocess
import time
import multiprocessing


def run_dense_reconstruction(parent_dir, image_path, sparse_path, enable_dense=True, max_image_size=3200, quality_mode=False, ultra_sharpness_mode=False, colmap_path='colmap'):
    """
    Run dense stereo reconstruction after sparse SfM

    Args:
        parent_dir: Base directory containing all processing outputs
        image_path: Path to source images
        sparse_path: Path to sparse reconstruction (sparse/0/)
        enable_dense: Whether to run dense reconstruction (can be disabled for faster processing)
        max_image_size: Maximum image dimension for dense reconstruction (higher = more detail but slower)
        quality_mode: Enable maximum quality settings (slower but best results)
        ultra_sharpness_mode: Enable maximum sharpness settings (slowest but production-grade results)
        colmap_path: Path to COLMAP executable (defaults to 'colmap' in PATH)

    Returns:
        Path to dense point cloud PLY file, or None if failed/disabled
    """

    if not enable_dense:
        print("Dense reconstruction disabled, skipping...")
        return None

    if not os.path.exists(sparse_path):
        print(f"ERROR: Sparse reconstruction not found at {sparse_path}")
        return None

    print("=" * 60)
    print("DENSE RECONSTRUCTION PIPELINE")
    print("=" * 60)
    print(f"This will generate millions of points instead of thousands")
    print(f"Max image size: {max_image_size}px (higher = more detail)")
    print(f"Using GPU acceleration if available")
    print("=" * 60)

    dense_workspace = os.path.join(parent_dir, 'dense')
    os.makedirs(dense_workspace, exist_ok=True)

    stereo_folder = os.path.join(dense_workspace, 'stereo')
    fused_ply_path = os.path.join(dense_workspace, 'fused.ply')

    total_start = time.time()

    # Step 1: Image undistortion (prepare images for dense matching)
    print("\n[1/3] Undistorting images for dense matching...")
    undist_start = time.time()

    undist_cmd = (
        f'"{colmap_path}" image_undistorter '
        f'--image_path "{image_path}" '
        f'--input_path "{sparse_path}" '
        f'--output_path "{dense_workspace}" '
        f'--output_type COLMAP '
        f'--max_image_size {max_image_size}'
    )

    result = subprocess.run(undist_cmd, shell=True, capture_output=True, text=True)
    undist_time = time.time() - undist_start

    if result.returncode != 0:
        print(f"ERROR: Image undistortion failed!")
        print(f"STDERR: {result.stderr}")
        return None

    print(f"✓ Image undistortion complete ({undist_time:.1f}s)")

    # Step 2: Patch match stereo (dense depth map computation)
    if quality_mode:
        print("\n[2/3] Computing dense depth maps with QUALITY MODE (this will take longer)...")
    else:
        print("\n[2/3] Computing dense depth maps (this may take several minutes)...")
    stereo_start = time.time()

    # Get number of available CPU cores for parallel processing
    num_threads = multiprocessing.cpu_count()

    # Adjust stereo parameters based on quality mode
    if ultra_sharpness_mode:
        # Ultra Sharpness mode: Maximum detail for production-grade results
        window_radius = 11  # Largest window = maximum context
        num_samples = 40    # Maximum samples = best accuracy
        num_iterations = 20 # Maximum iterations = best convergence
        filter = 0.3  # Very permissive
        print("  🔥 Using ULTRA SHARPNESS MODE: 11px window, 40 samples, 20 iterations")
    elif quality_mode:
        # Quality mode: Maximum detail settings
        window_radius = 7  # Larger window = more context (default 5)
        num_samples = 25   # More samples = better accuracy (default 15)
        num_iterations = 10  # More iterations = better convergence (default 5)
        filter = 0.5  # Lower = more permissive (default higher)
        print("  Using QUALITY MODE: 7px window, 25 samples, 10 iterations")
    else:
        # Balanced settings
        window_radius = 5
        num_samples = 15
        num_iterations = 5
        filter = 1.0

    stereo_cmd = (
        f'"{colmap_path}" patch_match_stereo '
        f'--workspace_path "{dense_workspace}" '
        f'--workspace_format COLMAP '
        f'--PatchMatchStereo.geom_consistency true '
        f'--PatchMatchStereo.gpu_index 0 '
        f'--PatchMatchStereo.depth_min 0.0 '
        f'--PatchMatchStereo.depth_max 100.0 '
        f'--PatchMatchStereo.window_radius {window_radius} '
        f'--PatchMatchStereo.num_samples {num_samples} '
        f'--PatchMatchStereo.num_iterations {num_iterations} '
        f'--PatchMatchStereo.filter_min_ncc 0.1 '
        f'--PatchMatchStereo.filter_min_triangulation_angle 1.0 '
        f'--PatchMatchStereo.filter_min_num_consistent 2 '
        f'--PatchMatchStereo.filter_geom_consistency_max_cost {filter} '
        f'--PatchMatchStereo.cache_size 32'
    )

    result = subprocess.run(stereo_cmd, shell=True, capture_output=True, text=True)
    stereo_time = time.time() - stereo_start

    if result.returncode != 0:
        print(f"WARNING: Stereo matching had issues (code {result.returncode})")
        print(f"STDERR: {result.stderr[:500]}")
        # Don't return - try fusion anyway, some depth maps may have succeeded
    else:
        print(f"✓ Depth map computation complete ({stereo_time:.1f}s)")

    # Step 3: Stereo fusion (merge depth maps into dense point cloud)
    if quality_mode:
        print("\n[3/3] Fusing depth maps with QUALITY MODE (maximum points)...")
    else:
        print("\n[3/3] Fusing depth maps into dense point cloud...")
    fusion_start = time.time()

    # Adjust fusion parameters based on quality mode
    if ultra_sharpness_mode:
        # Ultra Sharpness mode: Maximum permissive settings for production-grade point count
        min_num_pixels = 2  # Minimum pixels for maximum points
        max_reproj_error = 6.0  # Very permissive reprojection
        max_depth_error = 0.03  # Very permissive depth
        max_normal_error = 20  # Very permissive normal
        check_num_images = 3  # Minimum images for maximum points
        print("  🔥 Using ULTRA SHARPNESS MODE fusion: Maximum permissive thresholds")
    elif quality_mode:
        # Quality mode: More permissive settings for maximum point count
        min_num_pixels = 3  # Lower = more points (default 5)
        max_reproj_error = 4.0  # Higher = more points (default 2.0)
        max_depth_error = 0.02  # Higher = more points (default 0.01)
        max_normal_error = 15  # Higher = more points (default 10)
        check_num_images = 5  # Lower = more points (default 10)
        print("  Using QUALITY MODE fusion: More permissive thresholds for maximum points")
    else:
        # Balanced settings
        min_num_pixels = 5
        max_reproj_error = 2.0
        max_depth_error = 0.01
        max_normal_error = 10
        check_num_images = 10

    fusion_cmd = (
        f'"{colmap_path}" stereo_fusion '
        f'--workspace_path "{dense_workspace}" '
        f'--workspace_format COLMAP '
        f'--input_type geometric '
        f'--output_path "{fused_ply_path}" '
        f'--StereoFusion.min_num_pixels {min_num_pixels} '
        f'--StereoFusion.max_reproj_error {max_reproj_error} '
        f'--StereoFusion.max_depth_error {max_depth_error} '
        f'--StereoFusion.max_normal_error {max_normal_error} '
        f'--StereoFusion.check_num_images {check_num_images}'
    )

    result = subprocess.run(fusion_cmd, shell=True, capture_output=True, text=True)
    fusion_time = time.time() - fusion_start

    if result.returncode != 0:
        print(f"ERROR: Stereo fusion failed!")
        print(f"STDERR: {result.stderr}")
        return None

    print(f"✓ Dense fusion complete ({fusion_time:.1f}s)")

    total_time = time.time() - total_start

    # Verify output file exists and get point count
    if os.path.exists(fused_ply_path):
        file_size_mb = os.path.getsize(fused_ply_path) / (1024 * 1024)

        # Try to count points in PLY file
        try:
            with open(fused_ply_path, 'r') as f:
                for line in f:
                    if line.startswith('element vertex'):
                        point_count = int(line.split()[-1])
                        print("\n" + "=" * 60)
                        print(f"SUCCESS! Dense point cloud generated:")
                        print(f"  Points: {point_count:,}")
                        print(f"  File size: {file_size_mb:.1f} MB")
                        print(f"  Total time: {total_time:.1f}s")
                        print(f"  Location: {fused_ply_path}")
                        print("=" * 60)
                        break
        except Exception as e:
            print(f"Dense PLY created but couldn't read point count: {e}")

        return fused_ply_path
    else:
        print("ERROR: Dense PLY file was not created")
        return None


def run_poisson_reconstruction(dense_ply_path, output_path, depth=10, colmap_path='colmap'):
    """
    Optional: Run Poisson surface reconstruction on dense point cloud
    Creates a mesh from the point cloud

    Args:
        dense_ply_path: Path to dense point cloud
        output_path: Output path for mesh
        depth: Octree depth (higher = more detail, default 10)
        colmap_path: Path to COLMAP executable (defaults to 'colmap' in PATH)

    Returns:
        Path to mesh file or None if failed
    """

    if not os.path.exists(dense_ply_path):
        print(f"ERROR: Dense PLY not found at {dense_ply_path}")
        return None

    print(f"\n[Optional] Running Poisson reconstruction (depth={depth})...")

    poisson_cmd = (
        f'"{colmap_path}" poisson_mesher '
        f'--input_path "{dense_ply_path}" '
        f'--output_path "{output_path}" '
        f'--PoissonMeshing.depth {depth} '
        f'--PoissonMeshing.trim 7'
    )

    result = subprocess.run(poisson_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Poisson reconstruction failed: {result.stderr}")
        return None

    print(f"✓ Mesh generated: {output_path}")
    return output_path


if __name__ == "__main__":
    """
    Test the dense reconstruction module standalone
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run dense reconstruction on COLMAP sparse output")
    parser.add_argument('--parent_dir', required=True, help="Parent directory containing sparse/ folder")
    parser.add_argument('--image_path', required=True, help="Path to source images")
    parser.add_argument('--max_image_size', type=int, default=3200, help="Max image dimension (default: 3200)")
    parser.add_argument('--enable_dense', type=bool, default=True, help="Enable dense reconstruction")

    args = parser.parse_args()

    sparse_path = os.path.join(args.parent_dir, 'sparse', '0')

    result = run_dense_reconstruction(
        args.parent_dir,
        args.image_path,
        sparse_path,
        args.enable_dense,
        args.max_image_size
    )

    if result:
        print(f"\nDense reconstruction successful: {result}")
    else:
        print("\nDense reconstruction failed")
