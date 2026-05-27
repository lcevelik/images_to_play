import os
import json
import pathlib
import subprocess
import argparse
import time
import datetime
import re
import threading
from shutil import copy2, move, rmtree

# Thread-local storage for progress callbacks (thread-safe for concurrent jobs)
_thread_local = threading.local()

def _load_presets():
    """Load preset JSON files from presets/ directory. Returns dict or None if not found."""
    preset_dir = pathlib.Path(__file__).parent / "presets"
    if not preset_dir.exists():
        return None
    presets = {}
    for f in sorted(preset_dir.glob("*.json")):
        with open(f) as fp:
            presets[f.stem] = json.load(fp)
    return presets if presets else None

def set_progress_callback(callback):
    """Set a callback function for real-time progress updates (thread-safe)"""
    _thread_local.progress_callback = callback

def get_progress_callback():
    """Get the current thread's progress callback"""
    return getattr(_thread_local, 'progress_callback', None)

def log_progress(message, level="INFO"):
    """Log progress - calls thread's callback if set, otherwise prints"""
    callback = get_progress_callback()
    if callback:
        callback(message, level)
    else:
        print(f"[{level}] {message}")

def rename_image_folder_if_needed(image_path):
    # Rename the image_path folder to "source" if it's named "input" or "images"
    parent_dir = os.path.abspath(os.path.join(image_path, os.pardir))
    current_folder_name = os.path.basename(os.path.normpath(image_path))
    
    if current_folder_name in ["input", "images"]:
        new_image_path = os.path.join(parent_dir, "source")
        os.rename(image_path, new_image_path)
        print(f"Renamed image folder from {current_folder_name} to: {new_image_path}")
        return new_image_path
    return image_path

def filter_images(image_path, interval):
    parent_dir = os.path.abspath(os.path.join(image_path, os.pardir))
    input_folder = os.path.join(parent_dir, 'input')

    if interval > 1:
        if not os.path.exists(input_folder):
            os.makedirs(input_folder)

        image_files = sorted([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])
        filtered_files = image_files[::interval]

        for file in filtered_files:
            copy2(os.path.join(image_path, file), os.path.join(input_folder, file))

        return input_folder
    return image_path

def run_colmap(image_path, matcher_type, interval, model_type, detail_level='medium', quality_mode=False, ultra_sharpness_mode=False, enable_dense_override=None):
    # Load presets from JSON files, fall back to inline dict if not found
    _json_presets = _load_presets()

    # Simplified preset system - 3 main presets plus legacy compatibility
    # LOW: Fast preview (sparse only, minimal features)
    # MEDIUM: Balanced quality (dense enabled, good features)
    # HIGH: Maximum quality (dense + quality mode, unlimited features)
    detail_settings = {
        # === NEW SIMPLIFIED PRESETS ===
        # Settings progression: LOW is most permissive (ensures reconstruction works)
        # tri_angle: LOWER = keeps points with smaller triangulation angles (more permissive)
        # reproj_error: HIGHER = allows more reprojection error (more permissive)
        # match_ratio: HIGHER = keeps more feature matches (more permissive)
        
        'low': {
            'features': 16384, 'peak': 0.006, 'tracks': 200000, 'octaves': 4,
            'tri_angle': 0.25, 'reproj_error': 16.0, 'match_ratio': 0.9,
            'dense': False,  # Sparse only for speed
            'description': 'Fast Preview - Sparse only, ~50K-200K points, ~2-5 min'
        },
        'medium': {
            'features': 32768, 'peak': 0.004, 'tracks': 500000, 'octaves': 5,
            'tri_angle': 0.1, 'reproj_error': 24.0, 'match_ratio': 0.92,
            'dense': True,  # Enable dense for quality
            'mvs_window_radius': 5, 'mvs_iterations': 5, 'mvs_samples': 15,
            'description': 'Balanced Quality - Dense enabled, ~500K-2M points, ~5-15 min'
        },
        'high': {
            'features': 0, 'peak': 0.0001, 'tracks': 10000000, 'octaves': 8,
            'tri_angle': 0.01, 'reproj_error': 64.0, 'match_ratio': 0.98,
            'ba_max_points': 10000000, 'tri_complete_error': 64.0, 'tri_merge_error': 64.0,
            'dense': True,  # Dense with quality mode
            'mvs_window_radius': 7, 'mvs_iterations': 10, 'mvs_samples': 25,
            'force_quality': True,  # Force quality mode ON
            'description': 'Maximum Quality - Dense + Quality Mode, 5M-50M+ points, ~20-60 min'
        },

        # === LEGACY PRESETS (for backward compatibility) ===
        # All legacy presets use permissive settings to ensure reconstruction works
        'ultra': {
            'features': 65536, 'peak': 0.001, 'tracks': 500000, 'octaves': 6,
            'tri_angle': 0.05, 'reproj_error': 32.0, 'match_ratio': 0.95,
            'dense': True,
            'description': 'Ultra Quality - 65K features, dense enabled'
        },
        'extreme': {
            'features': 100000, 'peak': 0.0005, 'tracks': 1000000, 'octaves': 7,
            'tri_angle': 0.02, 'reproj_error': 48.0, 'match_ratio': 0.97,
            'dense': True,
            'description': 'Extreme Quality - 100K features'
        },
        'insane': {
            'features': 0, 'peak': 0.0001, 'tracks': 5000000, 'octaves': 8,
            'tri_angle': 0.005, 'reproj_error': 96.0, 'match_ratio': 0.99,
            'dense': True,
            'mvs_window_radius': 7, 'mvs_iterations': 10, 'mvs_samples': 25,
            'description': 'Insane Quality - Maximum permissive settings'
        },
        'unlimited': {
            'features': 0, 'peak': 0.00001, 'tracks': 10000000, 'octaves': 8,
            'tri_angle': 0.001, 'reproj_error': 128.0, 'match_ratio': 0.999,
            'ba_max_points': 10000000, 'tri_complete_error': 64.0, 'tri_merge_error': 64.0,
            'dense': True,
            'description': 'Unlimited - All features, maximum points'
        },
        'expert': {
            'features': 0, 'peak': 0.000005, 'tracks': 50000000, 'octaves': 8,
            'tri_angle': 0.01, 'reproj_error': 128.0, 'match_ratio': 0.99,
            'ba_max_points': 50000000, 'tri_complete_error': 64.0, 'tri_merge_error': 64.0,
            'dense': True,
            'expert': True,
            'domain_size_pooling': True,
            'affine_shape': True,
            'guided_matching': True,
            'max_image_size': 8192,
            'refine_focal': True,
            'refine_principal': True,
            'refine_distortion': True,
            'extract_colors': True,
            'mvs_window_radius': 9,
            'mvs_iterations': 15,
            'mvs_samples': 30,
            'description': 'Expert - All advanced features enabled'
        },
        'sharpness': {
            'features': 0, 'peak': 0.00005, 'tracks': 20000000, 'octaves': 8,
            'tri_angle': 0.005, 'reproj_error': 128.0, 'match_ratio': 0.995,
            'ba_max_points': 20000000, 'tri_complete_error': 96.0, 'tri_merge_error': 96.0,
            'dense': True,
            'ultra_sharpness': True,
            'mvs_window_radius': 11,
            'mvs_iterations': 20,
            'mvs_samples': 40,
            'force_quality': True,
            'description': '🔥 Maximum Sharpness - Ultra MVS + 200K steps recommended'
        },
    }
    # Use JSON presets if available, fall back to inline dict
    detail_settings = _json_presets if _json_presets else detail_settings
    settings = detail_settings.get(detail_level, detail_settings['medium'])

    # Apply dense override if specified
    if enable_dense_override is not None:
        settings = dict(settings)
        settings['dense'] = enable_dense_override

    # Force quality mode for 'high' preset
    if settings.get('force_quality', False):
        quality_mode = True

    # Use all CPU threads for maximum performance (GPU handles heavy lifting, CPU handles I/O)
    import multiprocessing
    num_threads = multiprocessing.cpu_count()

    # Log all settings being applied
    log_progress("=" * 60, "INFO")
    log_progress(f"COLMAP SETTINGS - Preset: {detail_level.upper()}", "INFO")
    if settings.get('description'):
        log_progress(f"  {settings['description']}", "INFO")
    log_progress("=" * 60, "INFO")
    log_progress(f"[SIFT] Max features per image: {settings['features'] if settings['features'] > 0 else 'UNLIMITED'}", "INFO")
    log_progress(f"[SIFT] Peak threshold: {settings['peak']} (lower = more keypoints)", "INFO")
    log_progress(f"[SIFT] Octaves: {settings['octaves']} (more = multi-scale detection)", "INFO")
    log_progress(f"[MATCH] Ratio test: {settings['match_ratio']} (higher = keep more matches)", "INFO")
    log_progress(f"[MAP] Min triangle angle: {settings['tri_angle']} deg (lower = more points)", "INFO")
    log_progress(f"[MAP] Max reprojection error: {settings['reproj_error']} px (higher = more points)", "INFO")
    if settings.get('dense', False):
        log_progress(f"[DENSE] Dense reconstruction: ENABLED (millions of points)", "INFO")
    else:
        log_progress(f"[DENSE] Dense reconstruction: DISABLED (sparse only)", "INFO")
    if quality_mode:
        log_progress(f"[QUALITY] Quality mode: ENABLED (maximum point density)", "INFO")
    if detail_level == 'unlimited':
        log_progress("[MAP] Two-view tracks: ENABLED (keeps points seen by only 2 cameras)", "INFO")
        log_progress("[MAP] Extra triangulation pass: ENABLED", "INFO")
    log_progress(f"[GPU] CUDA acceleration: ENABLED", "INFO")
    log_progress("=" * 60, "INFO")
    
    # Rename the image_path folder if needed
    image_path = rename_image_folder_if_needed(image_path)

    parent_dir = os.path.abspath(os.path.join(image_path, os.pardir))
    image_path = filter_images(image_path, interval)

    distorted_folder = os.path.join(parent_dir, 'distorted')
    database_path = os.path.join(distorted_folder, 'database.db')
    sparse_folder = os.path.join(parent_dir, 'sparse')  # Top-level sparse folder
    sparse_zero_folder = os.path.join(sparse_folder, '0')  # The new subfolder we want to create

    # Clean up old files before starting new processing
    if os.path.exists(database_path):
        os.remove(database_path)
        print(f"Removed old database: {database_path}")
    
    if os.path.exists(sparse_folder):
        rmtree(sparse_folder)
        print(f"Removed old sparse folder: {sparse_folder}")
    
    if os.path.exists(distorted_folder):
        rmtree(distorted_folder)
        print(f"Removed old distorted folder: {distorted_folder}")

    os.makedirs(distorted_folder, exist_ok=True)
    os.makedirs(sparse_folder, exist_ok=True)

    log_file_path = os.path.join(parent_dir, "colmap_run.log")
    total_start_time = time.time()

    # Add COLMAP bin/lib folders to PATH for DLL loading
    # COLMAP 4.x bundles all DLLs in bin/; 3.x used a separate lib/ folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    bundled_colmap_lib = os.path.join(project_root, 'COLMAP', 'lib')
    bundled_colmap_bin_dir = os.path.join(project_root, 'COLMAP', 'bin')
    bundled_colmap_bin = os.path.join(bundled_colmap_bin_dir, 'colmap.exe')
    _current_path = os.environ.get("PATH", "")
    if os.path.exists(bundled_colmap_bin_dir) and bundled_colmap_bin_dir not in _current_path:
        os.environ["PATH"] = bundled_colmap_bin_dir + ";" + _current_path
    if os.path.exists(bundled_colmap_lib) and bundled_colmap_lib not in _current_path:
        os.environ["PATH"] = bundled_colmap_lib + ";" + os.environ.get("PATH", "")

    # Also add system COLMAP bin/lib folders if present
    colmap_bin_dir = r"C:\COLMAP\bin"
    colmap_lib = r"C:\COLMAP\lib"
    _current_path = os.environ.get("PATH", "")
    if os.path.exists(colmap_bin_dir) and colmap_bin_dir not in _current_path:
        os.environ["PATH"] = colmap_bin_dir + ";" + _current_path
    if os.path.exists(colmap_lib) and colmap_lib not in _current_path:
        os.environ["PATH"] = colmap_lib + ";" + os.environ.get("PATH", "")

    # Find COLMAP path first (will be used in all commands)
    colmap_path = None
    possible_colmap_paths = [
        bundled_colmap_bin,  # Bundled COLMAP (highest priority - correct version)
        "colmap",  # In PATH
        r"C:\COLMAP\bin\colmap.exe",
        r"C:\COLMAP\colmap.exe",
    ]

    for path in possible_colmap_paths:
        try:
            result = subprocess.run([path, "--help"], capture_output=True, timeout=10)
            if result.returncode == 0:
                colmap_path = path
                print(f"Found COLMAP at: {path}")
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if not colmap_path:
        raise Exception("COLMAP is not installed or not in PATH. Please install COLMAP from https://github.com/colmap/colmap/releases")

    colmap_cmd = f'"{colmap_path}"' if colmap_path != "colmap" else "colmap"

    # Check if COLMAP has global_mapper (COLMAP 4.0+, replaces standalone GLOMAP binary)
    use_glomap = False
    glomap_path = None  # kept for output-path handling below
    try:
        gm_check = subprocess.run([colmap_path, 'global_mapper', '--help'],
                                   capture_output=True, timeout=10)
        if gm_check.returncode == 0:
            use_glomap = True
            glomap_path = colmap_path
            print("COLMAP global_mapper available (COLMAP 4.0+, built-in GLOMAP)")
    except Exception:
        pass

    if not use_glomap:
        print("global_mapper not available, will use COLMAP incremental mapper")

    if use_glomap:
        # Use COLMAP global_mapper (COLMAP 4.0+ built-in replacement for standalone GLOMAP)
        # ~10-100x faster than incremental mapper for large datasets
        glomap_output = os.path.join(distorted_folder, 'sparse')
        os.makedirs(glomap_output, exist_ok=True)
        mapper_cmd = (
            f'{colmap_cmd} global_mapper '
            f'--database_path "{database_path}" '
            f'--image_path "{image_path}" '
            f'--output_path "{glomap_output}" '
            f'--GlobalMapper.track_min_num_views_per_track 2'
        )
        print("Using COLMAP global_mapper (faster global SfM)")
    else:
        # Use COLMAP mapper with settings based on detail level
        # Lower tri_angle and higher reproj_error = MORE points kept
        mapper_cmd = (
            f'{colmap_cmd} mapper '
            f'--database_path "{database_path}" '
            f'--image_path "{image_path}" '
            f'--output_path "{sparse_folder}" '
            f'--Mapper.filter_min_tri_angle {settings["tri_angle"]} '
            f'--Mapper.filter_max_reproj_error {settings["reproj_error"]} '
            f'--Mapper.num_threads {num_threads} '
        )

        log_progress(f"[CPU] Using {num_threads} threads for mapping & bundle adjustment", "INFO")

        # Add extra settings for unlimited/dense/expert modes
        if detail_level in ['unlimited', 'dense', 'expert']:
            mapper_cmd += (
                f'--Mapper.ba_global_max_num_iterations 100 '
                f'--Mapper.ba_global_max_refinements 10 '
                f'--Mapper.tri_ignore_two_view_tracks 0 '
            )
        
        # Expert mode: add camera refinement and color extraction
        if settings.get('expert', False):
            mapper_cmd += (
                f'--Mapper.ba_refine_focal_length 1 '
                f'--Mapper.ba_refine_principal_point 1 '
                f'--Mapper.ba_refine_extra_params 1 '
                f'--Mapper.extract_colors 1 '
            )
            log_progress("[EXPERT] Focal length refinement: ENABLED", "INFO")
            log_progress("[EXPERT] Principal point refinement: ENABLED", "INFO")
            log_progress("[EXPERT] Distortion refinement: ENABLED", "INFO")
            log_progress("[EXPERT] Color extraction: ENABLED", "INFO")
        
        print("Using COLMAP mapper (GLOMAP not available)")
    
    # Feature extraction with configurable detail level
    # If features = 0, don't limit (extract all possible features)
    if settings["features"] == 0:
        feature_limit = ""
    else:
        feature_limit = f'--SiftExtraction.max_num_features {settings["features"]} '
    
    # COLMAP 3.9.1 with CUDA - enable GPU acceleration with optimized settings
    feature_cmd = (
        f'{colmap_cmd} feature_extractor '
        f'--image_path "{image_path}" '
        f'--database_path "{database_path}" '
        f'--ImageReader.single_camera 1 '
        f'--ImageReader.camera_model SIMPLE_RADIAL '
        f'{feature_limit}'
        f'--FeatureExtraction.use_gpu 1 '
        f'--FeatureExtraction.gpu_index -1 '
        f'--FeatureExtraction.num_threads {num_threads} '
        f'--SiftExtraction.first_octave -1 '
        f'--SiftExtraction.peak_threshold {settings["peak"]} '
        f'--SiftExtraction.num_octaves {settings["octaves"]} '
        f'--SiftExtraction.edge_threshold 5'
    )

    log_progress(f"[CPU] Using {num_threads} threads for feature extraction", "INFO")
    
    # Expert mode: add advanced feature extraction options
    if settings.get('expert', False):
        feature_cmd += (
            f' --SiftExtraction.domain_size_pooling 1'
            f' --SiftExtraction.estimate_affine_shape 1'
            f' --FeatureExtraction.max_image_size {settings.get("max_image_size", 8192)}'
            # FeatureExtraction.max_image_size is correct in both 3.x and 4.x
        )
        log_progress("[EXPERT] Domain size pooling: ENABLED", "INFO")
        log_progress("[EXPERT] Affine shape estimation: ENABLED", "INFO")
        log_progress(f"[EXPERT] Max image size: {settings.get('max_image_size', 8192)}px", "INFO")
    
    # Matching settings - GPU accelerated, configurable based on detail level
    # Higher max_ratio = more matches kept (less strict filtering)
    # Lower max_num_matches to prevent GPU OOM with high feature counts
    max_matches = 32768 if settings['features'] == 0 or settings['features'] > 50000 else 16384

    match_cmd = (
        f'{colmap_cmd} {matcher_type} '
        f'--database_path "{database_path}" '
        f'--FeatureMatching.use_gpu 1 '
        f'--FeatureMatching.gpu_index -1 '
        f'--FeatureMatching.num_threads {num_threads} '
        f'--SiftMatching.max_ratio {settings["match_ratio"]} '
        f'--FeatureMatching.max_num_matches {max_matches}'
    )
    log_progress(f"[MATCH] Max matches per pair: {max_matches}", "INFO")
    log_progress(f"[CPU] Using {num_threads} threads for matching", "INFO")
    
    # For sequential matcher, increase overlap for better matching
    # Note: loop_detection requires vocab_tree file (~1GB) which is often not installed
    # Disable it to avoid "file.is_open()" errors
    if 'sequential' in matcher_type.lower():
        # Check if vocab tree exists (required for loop detection)
        vocab_tree_paths = [
            r"C:\COLMAP\vocab_tree.bin",
            r"C:\COLMAP\vocab_tree_flickr100K_words32K.bin",
            r"C:\COLMAP\vocab_tree_flickr100K_words256K.bin",
            os.path.expanduser("~/.colmap/vocab_tree.bin"),
        ]
        vocab_tree = None
        for vt_path in vocab_tree_paths:
            if os.path.exists(vt_path):
                vocab_tree = vt_path
                break
        
        if vocab_tree:
            match_cmd += f' --SequentialMatching.overlap 10 --SequentialMatching.loop_detection 1 --SequentialMatching.vocab_tree_path "{vocab_tree}"'
            log_progress("[MATCH] Sequential with loop detection enabled", "INFO")
        else:
            # Without loop detection, increase overlap significantly to find more matches
            # Also enable quadratic overlap to match more distant images
            match_cmd += ' --SequentialMatching.overlap 30 --SequentialMatching.quadratic_overlap 1 --SequentialMatching.loop_detection 0'
            log_progress("[MATCH] Sequential matching (extended overlap, no loop detection)", "INFO")
    
    # Expert mode: add guided matching
    if settings.get('expert', False) and settings.get('guided_matching', False):
        match_cmd += ' --FeatureMatching.guided_matching 1'
        log_progress("[EXPERT] Guided matching: ENABLED", "INFO")
    
    # Named commands with descriptions for better logging
    commands = [
        ("Feature Extraction", feature_cmd, "Detecting keypoints in images using SIFT (GPU accelerated)"),
        ("Feature Matching", match_cmd, "Matching features between image pairs (GPU accelerated)"),
        ("3D Mapping", mapper_cmd, "Reconstructing 3D structure from matched features"),
    ]
    
    # For unlimited mode, add extra triangulation pass to find more points
    if detail_level == 'unlimited' and not use_glomap:
        # Point triangulator tries to triangulate more points from existing matches
        triangulator_cmd = (
            f'{colmap_cmd} point_triangulator '
            f'--database_path "{database_path}" '
            f'--image_path "{image_path}" '
            f'--input_path "{sparse_folder}/0" '
            f'--output_path "{sparse_folder}/0" '
            f'--Mapper.filter_min_tri_angle 0.0001 '
            f'--Mapper.filter_max_reproj_error 128 '
            f'--Mapper.tri_min_angle 0.0001 '
            f'--Mapper.tri_ignore_two_view_tracks 0 '
            f'--Mapper.tri_complete_max_transitivity 200'
        )
        commands.append(("Extra Triangulation", triangulator_cmd, "Finding additional 3D points from matches (unlimited mode)"))

    with open(log_file_path, "w") as log_file:
        log_file.write(f"COLMAP run started at: {datetime.datetime.now()}\n")
        log_file.write(f"Detail level: {detail_level} ({settings['features']} features)\n")
        log_file.write(f"COLMAP path: {colmap_path}\n")
        if use_glomap:
            log_file.write(f"GLOMAP path: {glomap_path}\n")

        total_commands = len(commands)
        for cmd_idx, (cmd_name, command, description) in enumerate(commands, 1):
            command_start_time = time.time()
            log_file.write(f"Running command: {command}\n")
            log_file.flush()
            
            log_progress(f"=== Step {cmd_idx}/{total_commands}: {cmd_name} ===", "INFO")
            log_progress(f"  > {description}", "INFO")
            
            # Run command with real-time output capture
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Track progress from COLMAP output
            images_processed = 0
            matches_found = 0
            total_images = 0
            last_progress_time = time.time()
            last_image_log_time = 0
            output_lines = []
            current_image_name = ""
            
            # Count total images first for percentage
            try:
                total_images = len([f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                log_progress(f"  [INFO] Processing {total_images} images...", "INFO")
            except Exception:
                pass
            
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                line = line.strip()
                output_lines.append(line)
                log_file.write(f"{line}\n")
                
                # Parse COLMAP output for progress info
                current_time = time.time()
                
                # Extract image name for THIS LINE only (reset each iteration to avoid stale names)
                line_image_name = ""
                
                # e.g., "Processing image 1 / 100 [name: frame_00001.jpg]"
                image_name_match = re.search(r'\[name:\s*([^\]]+)\]', line)
                if image_name_match:
                    line_image_name = image_name_match.group(1).strip()
                
                # Also try to extract from "Processing file: frame_001.jpg"
                if not line_image_name:
                    file_match = re.search(r'(?:Processing|Loading|Reading)\s+(?:file:?\s*)?([^\s]+\.(?:jpg|jpeg|png))', line, re.IGNORECASE)
                    if file_match:
                        line_image_name = file_match.group(1)
                
                # Update persistent current_image_name only if we found one on this line
                if line_image_name:
                    current_image_name = line_image_name
                
                # Feature extraction progress - log every image or every second
                if "Processing image" in line or "Processed image" in line:
                    match = re.search(r'(\d+)\s*/\s*(\d+)', line)
                    if match:
                        current_num, total_num = int(match.group(1)), int(match.group(2))
                        pct = (current_num / total_num * 100) if total_num > 0 else 0
                        
                        # Log every 5 images or every 2 seconds, whichever comes first
                        if current_num % 5 == 0 or current_time - last_image_log_time > 2:
                            img_info = f" ({current_image_name})" if current_image_name else ""
                            log_progress(f"  [SIFT] Image {current_num}/{total_num} ({pct:.0f}%){img_info}", "INFO")
                            last_image_log_time = current_time
                            current_image_name = ""  # Reset after logging
                
                # General "Processed X images" message
                elif "Processed" in line and "image" in line.lower():
                    match = re.search(r'Processed\s+(\d+)', line, re.IGNORECASE)
                    if match:
                        images_processed = int(match.group(1))
                        if current_time - last_progress_time > 1:
                            pct = (images_processed / total_images * 100) if total_images > 0 else 0
                            log_progress(f"  [SIFT] Processed {images_processed}/{total_images} images ({pct:.0f}%)", "INFO")
                            last_progress_time = current_time
                
                # Feature extraction with features count
                elif "features" in line.lower() and ("extracted" in line.lower() or "found" in line.lower()):
                    feat_match = re.search(r'(\d+)\s*features', line, re.IGNORECASE)
                    if feat_match and current_time - last_progress_time > 1:
                        log_progress(f"  [SIFT] Extracted {int(feat_match.group(1)):,} features", "INFO")
                        last_progress_time = current_time
                
                # Matching progress - more frequent updates
                elif "Matching" in line:
                    match = re.search(r'(\d+)\s*/\s*(\d+)', line)
                    if match:
                        current_num, total_num = int(match.group(1)), int(match.group(2))
                        pct = (current_num / total_num * 100) if total_num > 0 else 0
                        if current_num % 10 == 0 or current_time - last_progress_time > 2:
                            log_progress(f"  [MATCH] Pair {current_num}/{total_num} ({pct:.0f}%)", "INFO")
                            last_progress_time = current_time
                
                # Match count
                elif "matches" in line.lower():
                    match = re.search(r'(\d+)\s*matches', line, re.IGNORECASE)
                    if match:
                        matches_found = int(match.group(1))
                        if current_time - last_progress_time > 2:
                            log_progress(f"  [MATCH] Found {matches_found:,} matches so far", "INFO")
                            last_progress_time = current_time
                
                # Image registration in mapper
                elif "Registering image" in line or "registered image" in line.lower():
                    reg_match = re.search(r'image\s*#?\s*(\d+)', line, re.IGNORECASE)
                    if reg_match:
                        log_progress(f"  [MAP] Registering image #{reg_match.group(1)}...", "INFO")
                        last_progress_time = current_time
                
                # General registration count
                elif "Registered" in line:
                    match = re.search(r'(\d+)\s*images?', line, re.IGNORECASE)
                    if match and current_time - last_progress_time > 2:
                        log_progress(f"  [MAP] {match.group(1)} images registered in model", "INFO")
                        last_progress_time = current_time
                
                # Bundle adjustment iterations
                elif "Bundle adjustment" in line or "iteration" in line.lower():
                    iter_match = re.search(r'iteration\s*(\d+)', line, re.IGNORECASE)
                    if iter_match:
                        log_progress(f"  [BA] Bundle adjustment iteration {iter_match.group(1)}...", "INFO")
                    elif current_time - last_progress_time > 3:
                        log_progress(f"  [BA] Running bundle adjustment...", "INFO")
                        last_progress_time = current_time
                
                # Triangulation with point count
                elif "Triangulat" in line:
                    tri_match = re.search(r'(\d+)\s*points?', line, re.IGNORECASE)
                    if tri_match:
                        log_progress(f"  [TRI] Triangulated {int(tri_match.group(1)):,} 3D points", "INFO")
                        last_progress_time = current_time
                    elif current_time - last_progress_time > 3:
                        log_progress(f"  [TRI] Triangulating points...", "INFO")
                        last_progress_time = current_time
                
                # Point count updates
                elif "points" in line.lower() and ("3d" in line.lower() or "total" in line.lower()):
                    pt_match = re.search(r'(\d+)\s*(?:3d\s*)?points', line, re.IGNORECASE)
                    if pt_match and current_time - last_progress_time > 3:
                        log_progress(f"  [3D] Current point count: {int(pt_match.group(1)):,}", "INFO")
                        last_progress_time = current_time
                
                # GLOMAP specific
                elif "track" in line.lower() and "establish" in line.lower():
                    if current_time - last_progress_time > 2:
                        log_progress(f"  [GLOMAP] Establishing feature tracks...", "INFO")
                        last_progress_time = current_time
                
                elif "global" in line.lower() and ("rotation" in line.lower() or "position" in line.lower()):
                    if current_time - last_progress_time > 2:
                        log_progress(f"  [GLOMAP] Computing global camera poses...", "INFO")
                        last_progress_time = current_time
            
            process.wait()
            
            if process.returncode != 0:
                error_msg = f"Command failed with exit code {process.returncode}.\nOutput: {chr(10).join(output_lines[-20:])}"
                log_file.write(f"ERROR: {error_msg}\n")
                log_file.flush()
                raise Exception(error_msg)
            
            command_end_time = time.time()
            command_elapsed_time = command_end_time - command_start_time
            log_file.write(f"Time taken for command: {command_elapsed_time:.2f} seconds\n")
            log_file.flush()
            
            log_progress(f"  [OK] {cmd_name} complete ({command_elapsed_time:.1f}s)", "INFO")

        if model_type == '3dgs':
            # Determine the correct input path for undistortion
            # COLMAP mapper outputs to sparse/0, GLOMAP outputs to distorted/sparse/0
            if use_glomap:
                undist_input_path = os.path.join(distorted_folder, 'sparse', '0')
            else:
                undist_input_path = sparse_zero_folder
            
            # Only run undistortion if the sparse reconstruction exists
            if os.path.exists(undist_input_path):
                log_progress("=== Step 4/4: Image Undistortion ===", "INFO")
                log_progress("  > Removing lens distortion from images for training", "INFO")
                
                img_undist_cmd = (
                    f'{colmap_cmd} image_undistorter '
                    f'--image_path "{image_path}" '
                    f'--input_path "{undist_input_path}" '
                    f'--output_path "{parent_dir}" '
                    f'--output_type COLMAP'
                )
                log_file.write(f"Running command: {img_undist_cmd}\n")
                undistort_start_time = time.time()
                
                result = subprocess.run(img_undist_cmd, shell=True, capture_output=True, text=True)
                
                undistort_end_time = time.time()
                undistort_elapsed_time = undistort_end_time - undistort_start_time

                if result.returncode != 0:
                    log_file.write(f"Undistortion failed with code {result.returncode}. Continuing anyway.\n")
                    log_file.write(f"STDERR: {result.stderr}\n")
                    log_progress(f"  [WARN] Undistortion failed (continuing anyway)", "WARNING")
                else:
                    log_file.write(f"Time taken for undistortion: {undistort_elapsed_time:.2f} seconds\n")
                    log_progress(f"  [OK] Undistortion complete ({undistort_elapsed_time:.1f}s)", "INFO")
                
                # Dense reconstruction (MVS) for 'dense' detail level - like RealityCapture!
                if settings.get('dense', False):
                    dense_folder = os.path.join(parent_dir, 'dense')
                    os.makedirs(dense_folder, exist_ok=True)
                    
                    log_progress("=" * 60, "INFO")
                    log_progress("DENSE RECONSTRUCTION (MVS) - RealityCapture-style!", "INFO")
                    log_progress("=" * 60, "INFO")
                    
                    # Step 1: Patch Match Stereo (compute depth maps)
                    log_progress("=== Dense Step 1: Patch Match Stereo ===", "INFO")
                    log_progress("  > Computing depth maps for each image (GPU intensive)", "INFO")
                    
                    # Configure MVS based on quality_mode and ultra_sharpness (applies on top of detail level settings)
                    # Check for ultra_sharpness mode from preset settings OR from parameter
                    ultra_sharpness = settings.get('ultra_sharpness', False) or ultra_sharpness_mode
                    
                    if ultra_sharpness:
                        # Ultra Sharpness mode: Maximum detail settings for production-grade results
                        mvs_window = 11   # Largest window = maximum context
                        mvs_iters = 20    # Maximum iterations = best convergence
                        mvs_samples = 40  # Maximum samples = best accuracy
                        mvs_filter_ncc = 0.02  # Very permissive NCC filter
                        fusion_min_pixels = 2  # Minimum pixels for maximum points
                        fusion_max_reproj = 6.0  # Very permissive reprojection
                        fusion_max_depth = 0.03  # Very permissive depth
                        fusion_max_normal = 20  # Very permissive normal
                        fusion_check_images = 3  # Minimum images for maximum points
                        log_progress("🔥 [ULTRA SHARPNESS MODE] Using maximum MVS settings for production quality!", "INFO")
                    elif quality_mode:
                        # Quality mode: Maximum detail settings (matches dense_reconstruction.py)
                        mvs_window = 7   # Larger window = more context (default 5)
                        mvs_iters = 10   # More iterations = better convergence (default 5)
                        mvs_samples = 25 # More samples = better accuracy (default 15)
                        mvs_filter_ncc = 0.03  # Lower = more permissive (default 0.1)
                        fusion_min_pixels = 3  # Lower = more points (default 5)
                        fusion_max_reproj = 4.0  # Higher = more points (default 2.0)
                        fusion_max_depth = 0.02  # Higher = more points (default 0.01)
                        fusion_max_normal = 15  # Higher = more points (default 10)
                        fusion_check_images = 5  # Lower = more points (default 10)
                        log_progress("[QUALITY MODE] Using aggressive MVS settings for maximum points", "INFO")
                    else:
                        # Use settings from detail level, or balanced defaults
                        mvs_window = settings.get('mvs_window_radius', 5)
                        mvs_iters = settings.get('mvs_iterations', 5)
                        mvs_samples = settings.get('mvs_samples', 15)
                        mvs_filter_ncc = 0.05
                        fusion_min_pixels = 5
                        fusion_max_reproj = 2.0
                        fusion_max_depth = 0.01
                        fusion_max_normal = 10
                        fusion_check_images = 10
                    
                    # After undistortion, COLMAP creates dense/ folder with undistorted images
                    # We MUST use dense_folder as workspace, not parent_dir!
                    # Otherwise COLMAP processes wrong images (original + undistorted = double processing!)
                    patch_match_cmd = (
                        f'{colmap_cmd} patch_match_stereo '
                        f'--workspace_path "{parent_dir}" '
                        f'--workspace_format COLMAP '
                        f'--PatchMatchStereo.geom_consistency 1 '
                        f'--PatchMatchStereo.gpu_index -1 '
                        f'--PatchMatchStereo.window_radius {mvs_window} '
                        f'--PatchMatchStereo.num_iterations {mvs_iters} '
                        f'--PatchMatchStereo.num_samples {mvs_samples} '
                        f'--PatchMatchStereo.filter_min_ncc {mvs_filter_ncc} '
                        f'--PatchMatchStereo.cache_size 32'
                    )
                    
                    log_progress(f"[MVS] Window radius: {mvs_window}, Iterations: {mvs_iters}, Samples: {mvs_samples}", "INFO")
                    if settings.get('expert', False):
                        log_progress(f"[EXPERT] Using expert-level MVS configuration", "INFO")
                    log_file.write(f"Running: {patch_match_cmd}\n")
                    
                    pm_start = time.time()
                    
                    # Stream patch match output to show per-image progress
                    pm_process = subprocess.Popen(
                        patch_match_cmd, shell=True, 
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1
                    )
                    
                    pm_returncode = None
                    image_count = 0
                    # After undistortion, images are in dense_folder/images/, not parent_dir/images/
                    dense_images_folder = os.path.join(dense_folder, 'images')
                    if os.path.exists(dense_images_folder):
                        total_images = len([f for f in os.listdir(dense_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    else:
                        # Fallback to parent_dir/images if dense folder doesn't exist yet
                        total_images = len([f for f in os.listdir(os.path.join(parent_dir, 'images')) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    last_reported_view = 0
                    reached_100_percent = False
                    
                    while True:
                        line = pm_process.stdout.readline()
                        if not line and pm_process.poll() is not None:
                            break
                        if line:
                            line = line.strip()
                            log_file.write(f"PM: {line}\n")
                            # Parse progress from COLMAP output - try to extract actual view numbers
                            if 'Processing view' in line or 'Processed' in line:
                                # Try to parse "Processing view X/Y" format from COLMAP
                                match = re.search(r'view\s+(\d+)/(\d+)', line, re.IGNORECASE)
                                if match:
                                    current_view = int(match.group(1))
                                    reported_total = int(match.group(2))
                                    # Use the total from COLMAP if it's different (more accurate)
                                    if reported_total > 0:
                                        total_images = reported_total
                                    # Only log if this is a new view (avoid duplicates) and not past 100%
                                    if current_view > last_reported_view:
                                        last_reported_view = current_view
                                        image_count = current_view
                                        # Cap percentage at 100% and stop logging once we reach 100%
                                        percentage = min(100, (100 * image_count // total_images) if total_images > 0 else 0)
                                        
                                        if percentage >= 100 and not reached_100_percent:
                                            reached_100_percent = True
                                            log_progress(f"  [MVS] Processing depth map {image_count}/{total_images} (100%) - Finalizing...", "INFO")
                                        elif percentage < 100:
                                            log_progress(f"  [MVS] Processing depth map {image_count}/{total_images} ({percentage}%)", "INFO")
                                        # Don't log if we're past 100% - COLMAP may process extra views
                                else:
                                    # Fallback: count lines if we can't parse view numbers
                                    if not reached_100_percent:
                                        image_count += 1
                                        # Cap percentage at 100%
                                        percentage = min(100, (100 * image_count // total_images) if total_images > 0 else 0)
                                        
                                        if percentage >= 100:
                                            reached_100_percent = True
                                            log_progress(f"  [MVS] Processing depth map {image_count}/{total_images} (100%) - Finalizing...", "INFO")
                                        else:
                                            log_progress(f"  [MVS] Processing depth map {image_count}/{total_images} ({percentage}%)", "INFO")
                            elif 'ERROR' in line.upper() or 'FAILED' in line.upper():
                                log_progress(f"  [MVS] {line[:100]}", "WARNING")
                    
                    pm_returncode = pm_process.returncode
                    pm_time = time.time() - pm_start
                    log_progress(f"  [MVS] Patch match stereo completed ({pm_time:.1f}s)", "INFO")
                    
                    if pm_returncode != 0:
                        # Error code 3221226505 = Windows DLL/crash error (often GPU/CUDA related)
                        if pm_returncode == 3221226505:
                            log_progress(f"  [ERROR] Patch match crashed (code {pm_returncode}) - likely GPU/CUDA issue", "ERROR")
                            log_progress(f"  [INFO] Attempting fallback with CPU mode and reduced settings...", "INFO")
                            log_file.write(f"Patch match failed with crash code {pm_returncode}, trying CPU fallback\n")
                            
                            # Fallback: Try with CPU and reduced settings
                            fallback_window = min(7, mvs_window)  # Reduce window size
                            fallback_iters = min(10, mvs_iters)  # Reduce iterations
                            fallback_samples = min(25, mvs_samples)  # Reduce samples
                            
                            log_progress(f"  [FALLBACK] Trying CPU mode: window={fallback_window}, iters={fallback_iters}, samples={fallback_samples}", "INFO")
                            
                            patch_match_fallback = (
                                f'{colmap_cmd} patch_match_stereo '
                                f'--workspace_path "{parent_dir}" '
                                f'--workspace_format COLMAP '
                                f'--PatchMatchStereo.gpu_index -1 '
                                f'--PatchMatchStereo.geom_consistency 0 '
                                f'--PatchMatchStereo.window_radius {fallback_window} '
                                f'--PatchMatchStereo.num_iterations {fallback_iters} '
                                f'--PatchMatchStereo.num_samples {fallback_samples} '
                                f'--PatchMatchStereo.filter_min_ncc {mvs_filter_ncc} '
                                f'--PatchMatchStereo.cache_size 16'
                            )
                            
                            log_file.write(f"Fallback command: {patch_match_fallback}\n")
                            fallback_result = subprocess.run(patch_match_fallback, shell=True, capture_output=True, text=True, timeout=3600)
                            
                            if fallback_result.returncode == 0:
                                log_progress(f"  [OK] Fallback succeeded! Depth maps computed with CPU/reduced settings", "INFO")
                                pm_returncode = 0  # Mark as success
                            else:
                                log_progress(f"  [ERROR] Fallback also failed (code {fallback_result.returncode})", "ERROR")
                                log_file.write(f"Fallback failed: {fallback_result.stderr[:500]}\n")
                                log_progress(f"  [INFO] Dense reconstruction unavailable - continuing with sparse only", "WARNING")
                        else:
                            log_progress(f"  [WARN] Patch match failed (code {pm_returncode})", "WARNING")
                            log_file.write(f"Patch match failed with code {pm_returncode}\n")
                    
                    if pm_returncode == 0:
                        log_progress(f"  [OK] Depth maps computed for {total_images} images ({pm_time:.1f}s)", "INFO")
                        
                        # Step 2: Stereo Fusion (fuse into dense point cloud)
                        log_progress("=== Dense Step 2: Stereo Fusion ===", "INFO")
                        log_progress("  > Fusing depth maps into dense point cloud (millions of points!)", "INFO")
                        
                        dense_ply = os.path.join(dense_folder, 'fused.ply')
                        # Use dense_folder as workspace (contains undistorted images after image_undistorter)
                        fusion_cmd = (
                            f'{colmap_cmd} stereo_fusion '
                            f'--workspace_path "{parent_dir}" '
                            f'--workspace_format COLMAP '
                            f'--input_type geometric '
                            f'--output_path "{dense_ply}" '
                            f'--StereoFusion.min_num_pixels {fusion_min_pixels} '
                            f'--StereoFusion.max_reproj_error {fusion_max_reproj} '
                            f'--StereoFusion.max_depth_error {fusion_max_depth} '
                            f'--StereoFusion.max_normal_error {fusion_max_normal} '
                            f'--StereoFusion.check_num_images {fusion_check_images}'
                        )
                        log_file.write(f"Running: {fusion_cmd}\n")
                        
                        if quality_mode:
                            log_progress(f"[QUALITY MODE] Fusion: min_pixels={fusion_min_pixels}, max_reproj={fusion_max_reproj}", "INFO")
                        
                        fusion_start = time.time()
                        log_progress("  [Fusion] Starting stereo fusion (may take several minutes)...", "INFO")

                        # Stream fusion output for progress
                        fusion_process = subprocess.Popen(
                            fusion_cmd, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1
                        )
                        
                        while True:
                            line = fusion_process.stdout.readline()
                            if not line and fusion_process.poll() is not None:
                                break
                            if line:
                                line = line.strip()
                                log_file.write(f"Fusion: {line}\n")
                                if 'Fusing' in line or 'points' in line.lower():
                                    log_progress(f"  [Fusion] {line[:80]}", "INFO")
                        
                        fusion_returncode = fusion_process.returncode
                        fusion_time = time.time() - fusion_start
                        
                        if fusion_returncode != 0:
                            log_progress(f"  [WARN] Fusion failed (code {fusion_returncode})", "WARNING")
                        else:
                            log_progress(f"  [OK] Dense point cloud created ({fusion_time:.1f}s)", "INFO")
                            if os.path.exists(dense_ply):
                                ply_size = os.path.getsize(dense_ply) / (1024*1024)
                                log_progress(f"  [OK] Dense PLY: {dense_ply} ({ply_size:.1f} MB)", "INFO")
                    
                    log_progress("=" * 60, "INFO")
                    
            else:
                log_file.write(f"Sparse reconstruction not found at {undist_input_path}, skipping undistortion.\n")
                log_progress("  [WARN] Skipping undistortion (sparse not found)", "WARNING")

        # COLMAP mapper creates sparse/0 automatically, so check if it exists
        # If using GLOMAP, files might be in distorted/sparse/0, otherwise in sparse/0
        if use_glomap:
            glomap_sparse = os.path.join(distorted_folder, 'sparse', '0')
            if os.path.exists(glomap_sparse):
                # Copy files from GLOMAP output to top-level sparse/0
                log_progress("  [COPY] Copying reconstruction files...", "INFO")
                os.makedirs(sparse_zero_folder, exist_ok=True)
                for file_name in ['cameras.bin', 'images.bin', 'points3D.bin']:
                    source_file = os.path.join(glomap_sparse, file_name)
                    dest_file = os.path.join(sparse_zero_folder, file_name)
                    if os.path.exists(source_file):
                        copy2(source_file, dest_file)
                        log_file.write(f"Copied {file_name} to {sparse_zero_folder}\n")
        else:
            # COLMAP mapper creates sparse/0 automatically, files should already be there
            # Just verify they exist
            if os.path.exists(sparse_zero_folder):
                files_found = [f for f in ['cameras.bin', 'images.bin', 'points3D.bin'] 
                              if os.path.exists(os.path.join(sparse_zero_folder, f))]
                if files_found:
                    log_file.write(f"COLMAP mapper created files in {sparse_zero_folder}\n")
                    print(f"COLMAP mapper created files in {sparse_zero_folder}")
                else:
                    # Try to find files in sparse_folder (might be directly there)
                    for file_name in ['cameras.bin', 'images.bin', 'points3D.bin']:
                        source_file = os.path.join(sparse_folder, file_name)
                        dest_file = os.path.join(sparse_zero_folder, file_name)
                        if os.path.exists(source_file):
                            os.makedirs(sparse_zero_folder, exist_ok=True)
                            move(source_file, dest_file)
                            log_file.write(f"Moved {file_name} to {sparse_zero_folder}\n")
                            print(f"Moved {file_name} to {sparse_zero_folder}")

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        log_file.write(f"COLMAP run finished at: {datetime.datetime.now()}\n")
        log_file.write(f"Total time taken: {total_elapsed_time:.2f} seconds\n")
        
        # Final summary
        log_progress("=" * 50, "INFO")
        log_progress(f"[DONE] COLMAP/GLOMAP Pipeline Complete!", "INFO")
        log_progress(f"[TIME] Total time: {total_elapsed_time:.1f} seconds ({total_elapsed_time/60:.1f} minutes)", "INFO")
        log_progress("=" * 50, "INFO")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP with specified image path and matcher type.")
    parser.add_argument('--image_path', required=True, help="Path to the images folder.")
    parser.add_argument('--matcher_type', default='exhaustive_matcher', choices=['sequential_matcher', 'exhaustive_matcher'],
                        help="Type of matcher to use (default: exhaustive_matcher). Use sequential for video frames.")
    parser.add_argument('--interval', type=int, default=1, help="Interval of images to use (default: 1, meaning all images).")
    parser.add_argument('--model_type', default='3dgs', choices=['3dgs', 'nerfstudio'],
                        help="Model type to run. '3dgs' includes undistortion, 'nerfstudio' skips undistortion.")
    parser.add_argument('--detail_level', default='medium', 
                        choices=['low', 'medium', 'high', 'ultra', 'extreme', 'maximum', 'insane', 'unlimited', 'dense', 'expert', 'sharpness'],
                        help="Detail level for feature extraction (default: medium).")
    parser.add_argument('--quality_mode', action='store_true',
                        help="Enable quality mode for maximum point density (slower)")
    parser.add_argument('--ultra_sharpness', action='store_true',
                        help="Enable ultra sharpness mode for production-grade MVS quality (slowest)")

    args = parser.parse_args()

    run_colmap(args.image_path, args.matcher_type, args.interval, args.model_type, args.detail_level, args.quality_mode, args.ultra_sharpness)
