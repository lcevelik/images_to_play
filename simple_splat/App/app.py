import os
import re
import sys
import uuid
import zipfile
import shutil

# Force UTF-8 output so emoji/unicode in log messages don't crash on Windows cp1252 console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from flask import Flask, request, jsonify, render_template, send_file, Response, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
import subprocess
import threading
from pathlib import Path
import json
from datetime import datetime
from collections import deque
import time
import numpy as np

# Image processing constants
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def resize_images_for_colmap(image_folder, max_size=3840):
    """Downscale images larger than max_size to save COLMAP time.
    4K images slow COLMAP 16x vs 2K. Returns count of resized images."""
    from PIL import Image
    resized = 0
    for fname in os.listdir(image_folder):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue
        fpath = os.path.join(image_folder, fname)
        try:
            img = Image.open(fpath)
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                img.save(fpath, quality=95)
                resized += 1
        except Exception:
            continue
    return resized

def filter_blurry_images(image_folder, blur_threshold=100.0):
    """Detect blurry images using Laplacian variance. Returns list of (filename, score).
    Does NOT delete images — just reports for logging."""
    try:
        import cv2
    except ImportError:
        return []
    blurry = []
    for fname in os.listdir(image_folder):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue
        fpath = os.path.join(image_folder, fname)
        try:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            score = cv2.Laplacian(img, cv2.CV_64F).var()
            if score < blur_threshold:
                blurry.append((fname, score))
        except Exception:
            continue
    return blurry

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSING_FOLDER'] = 'processing'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'zip', 'mp4', 'mov', 'avi', 'mkv', 'webm'}

# Memory management settings
MAX_CONCURRENT_JOBS = 3  # Limit concurrent processing jobs to prevent memory exhaustion
JOB_RETENTION_HOURS = 1  # Keep completed jobs in memory for 1 hour
AUTO_CLEANUP_HOURS = 24  # Auto-cleanup job folders after 24 hours

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSING_FOLDER'], exist_ok=True)

# ML-Sharp availability (checked on startup)
mlsharp_available = False
mlsharp_version = None

def check_mlsharp_availability():
    """Check if ml-sharp is installed and get version"""
    global mlsharp_available, mlsharp_version
    try:
        result = subprocess.run(['sharp', '--help'], capture_output=True, text=True, timeout=30)
        if result.returncode in (0, 2):  # --help exits 0 or 2 depending on CLI framework
            mlsharp_available = True
            mlsharp_version = "ml-sharp 0.1"
            print(f"ML-Sharp detected: {mlsharp_version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ML-Sharp not found - feature disabled")
    return False

# Store processing status
processing_status = {}
job_start_times = {}  # Track when jobs were created

# GPU assignment tracking — maps job_id -> gpu_index
_gpu_lock = threading.Lock()
_job_gpu = {}  # job_id -> int gpu index

def _detect_gpu_count():
    """Return number of NVIDIA GPUs via nvidia-smi, default 1."""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            count = len([l for l in r.stdout.strip().splitlines() if l.strip()])
            return max(1, count)
    except Exception:
        pass
    return 1

GPU_COUNT = _detect_gpu_count()

def assign_gpu(job_id):
    """Pick the least-loaded GPU index for a new job and record it."""
    with _gpu_lock:
        in_use = list(_job_gpu.values())
        # Count how many active jobs are on each GPU
        counts = [in_use.count(i) for i in range(GPU_COUNT)]
        gpu = counts.index(min(counts))  # pick GPU with fewest active jobs
        _job_gpu[job_id] = gpu
        return gpu

def release_gpu(job_id):
    with _gpu_lock:
        _job_gpu.pop(job_id, None)

def gpu_env(gpu_index):
    """Return env dict that pins CUDA/Vulkan to the given GPU index."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    return env

# Log buffer for real-time logging
log_buffer = deque(maxlen=500)  # Keep last 500 log entries
log_lock = threading.Lock()

# Thread-local storage for current job ID (allows add_log to track per-job logs)
_current_job = threading.local()

# Background cleanup thread
cleanup_thread = None
cleanup_stop_event = threading.Event()

def set_current_job(job_id):
    """Set the current job ID for this thread (enables job-specific logging)"""
    _current_job.job_id = job_id

def get_current_job():
    """Get the current job ID for this thread"""
    return getattr(_current_job, 'job_id', None)

def transition_stage(processing_status, job_id, new_stage):
    """Transition to a new pipeline stage, recording elapsed time for the previous one."""
    ps = processing_status[job_id]
    old_stage = ps.get('stage')
    now = time.time()
    # Close out the old stage
    if old_stage and old_stage in ps.get('stages', {}):
        stage_info = ps['stages'][old_stage]
        if stage_info['status'] == 'running' and stage_info.get('started_at'):
            stage_info['elapsed'] = now - stage_info['started_at']
            stage_info['status'] = 'complete'
    # Start the new stage
    if new_stage in ps.get('stages', {}):
        ps['stages'][new_stage]['status'] = 'running'
        ps['stages'][new_stage]['started_at'] = now
    ps['stage'] = new_stage

def add_log(message, level="INFO"):
    """Add a log entry to the global buffer and job-specific buffer if available"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    with log_lock:
        log_buffer.append(entry)
        # Also add to job-specific logs if a job is active
        job_id = get_current_job()
        if job_id and job_id in processing_status:
            if 'logs' not in processing_status[job_id]:
                processing_status[job_id]['logs'] = deque(maxlen=200)
            processing_status[job_id]['logs'].append(entry)
    print(entry)  # Also print to console

def get_logs():
    """Get all logs from buffer"""
    with log_lock:
        return list(log_buffer)

def get_job_logs(job_id):
    """Get logs for a specific job"""
    with log_lock:
        if job_id in processing_status and 'logs' in processing_status[job_id]:
            return list(processing_status[job_id]['logs'])
        return []

def cleanup_old_jobs(max_age_hours=24):
    """Clean up processing folders older than max_age_hours"""
    processing_folder = app.config['PROCESSING_FOLDER']
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for job_folder in os.listdir(processing_folder):
        job_path = os.path.join(processing_folder, job_folder)
        if os.path.isdir(job_path):
            folder_age = current_time - os.path.getmtime(job_path)
            if folder_age > max_age_seconds:
                try:
                    shutil.rmtree(job_path)
                    print(f"Cleaned up old job: {job_folder}")
                except Exception as e:
                    print(f"Error cleaning up {job_folder}: {e}")

def cleanup_all_jobs():
    """Clean up all processing folders"""
    processing_folder = app.config['PROCESSING_FOLDER']

    # Check if processing folder exists
    if not os.path.exists(processing_folder):
        print("Processing folder does not exist, nothing to clean up")
        return

    # Clean up job folders
    for job_folder in os.listdir(processing_folder):
        job_path = os.path.join(processing_folder, job_folder)
        if os.path.isdir(job_path):
            try:
                shutil.rmtree(job_path)
                print(f"Cleaned up job: {job_folder}")
            except Exception as e:
                print(f"Error cleaning up {job_folder}: {e}")

    # Clear status dicts with proper locking
    with log_lock:
        processing_status.clear()
        job_start_times.clear()

def get_active_jobs_count():
    """Get count of currently processing jobs (excludes fast ML-Sharp jobs)"""
    return sum(
        1 for status in processing_status.values()
        if status.get('status') == 'processing'
        and status.get('method') != 'mlsharp'  # ML-Sharp is fast, don't count toward limit
    )

def cleanup_old_completed_jobs():
    """Clean up completed jobs from memory after retention period"""
    current_time = time.time()
    jobs_to_remove = []

    with log_lock:
        for job_id, start_time in list(job_start_times.items()):
            if job_id in processing_status:
                job_status = processing_status[job_id].get('status')
                age_hours = (current_time - start_time) / 3600

                # Remove completed/error jobs older than retention period from memory
                if job_status in ['completed', 'error', 'cancelled'] and age_hours > JOB_RETENTION_HOURS:
                    jobs_to_remove.append((job_id, age_hours))

    for job_id, age_hours in jobs_to_remove:
        processing_status.pop(job_id, None)
        job_start_times.pop(job_id, None)
        add_log(f"Cleaned up old job {job_id[:8]} from memory (age: {age_hours:.1f}h)", "INFO")

def background_cleanup_worker():
    """Background thread that periodically cleans up old jobs"""
    while not cleanup_stop_event.is_set():
        try:
            # Clean up old job folders every hour
            cleanup_old_jobs(max_age_hours=AUTO_CLEANUP_HOURS)

            # Clean up old completed jobs from memory
            cleanup_old_completed_jobs()

        except Exception as e:
            print(f"Background cleanup error: {e}")

        # Sleep for 1 hour (check every 60 seconds for stop event)
        for _ in range(60):
            if cleanup_stop_event.is_set():
                break
            time.sleep(60)

def start_background_cleanup():
    """Start the background cleanup thread"""
    global cleanup_thread
    if cleanup_thread is None or not cleanup_thread.is_alive():
        cleanup_stop_event.clear()
        cleanup_thread = threading.Thread(target=background_cleanup_worker, daemon=True)
        cleanup_thread.start()
        print("Background cleanup thread started")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_video_file(filename):
    """Check if file is a video"""
    video_extensions = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def extract_frames_from_video(video_path, output_folder, frame_interval=10, max_frames=1000):
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        output_folder: Folder to save extracted frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
    
    Returns:
        Number of frames extracted
    """
    try:
        import cv2
        
        add_log(f"Extracting frames from video: {video_path}", "INFO")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            add_log(f"Could not open video: {video_path}", "ERROR")
            return 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        add_log(f"Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds", "INFO")
        
        # Calculate optimal frame interval if we have too many frames
        if total_frames / frame_interval > max_frames:
            frame_interval = total_frames // max_frames
            add_log(f"Adjusted frame interval to {frame_interval} (max {max_frames} frames)", "INFO")
        
        os.makedirs(output_folder, exist_ok=True)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{extracted_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_count += 1
                
                if extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        add_log(f"Extracted {extracted_count} frames from video", "INFO")
        return extracted_count
        
    except ImportError:
        add_log("OpenCV not installed. Install with: pip install opencv-python", "ERROR")
        return 0
    except Exception as e:
        add_log(f"Error extracting frames: {e}", "ERROR")
        return 0

def extract_zip(zip_path, extract_to):
    """Extract zip file to directory"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def process_images_async(job_id, image_path, preset='medium', matcher_type='exhaustive_matcher', interval=1, advanced_settings=None, quality_scale=1.0, trainer='brush'):
    """Process images using COLMAP/GLOMAP pipeline with optional dense reconstruction"""
    try:
        # Preset configuration - maps simple presets to detailed settings
        preset_configs = {
            'low': {
                'detail_level': 'low',
                'training_steps': 3000,
                'enable_dense': False,
                'max_image_size': 1600,
                'quality_mode': False,
                'description': 'Fast Preview - Sparse only, ~50K-200K points, ~2-5 min'
            },
            'medium': {
                'detail_level': 'medium',
                'training_steps': 10000,
                'enable_dense': True,
                'max_image_size': 3200,
                'quality_mode': False,
                'description': 'Balanced Quality - Dense enabled, ~500K-2M points, ~5-15 min'
            },
            'high': {
                'detail_level': 'high',
                'training_steps': 30000,
                'enable_dense': True,
                'max_image_size': 4800,
                'quality_mode': True,
                'description': 'Maximum Quality - Dense + Quality Mode, 5M-50M+ points, ~20-60 min'
            },
            'ultra': {
                'detail_level': 'ultra',
                'training_steps': 50000,
                'enable_dense': True,
                'max_image_size': 4800,
                'quality_mode': True,
                'description': 'Ultra Quality - 65K features, dense enabled'
            },
            'extreme': {
                'detail_level': 'extreme',
                'training_steps': 80000,
                'enable_dense': True,
                'max_image_size': 4800,
                'quality_mode': True,
                'description': 'Extreme Quality - 100K features'
            },
            'insane': {
                'detail_level': 'insane',
                'training_steps': 150000,
                'enable_dense': True,
                'max_image_size': 4800,
                'quality_mode': True,
                'description': 'Insane Quality - Maximum permissive settings'
            },
            'unlimited': {
                'detail_level': 'unlimited',
                'training_steps': 200000,
                'enable_dense': True,
                'max_image_size': 4800,
                'quality_mode': True,
                'description': 'Unlimited - All features, maximum points'
            },
            'expert': {
                'detail_level': 'expert',
                'training_steps': 200000,
                'enable_dense': True,
                'max_image_size': 8192,
                'quality_mode': True,
                'description': 'Expert - All advanced features enabled'
            },
            'sharpness': {
                'detail_level': 'sharpness',
                'training_steps': 200000,
                'enable_dense': True,
                'max_image_size': 4800,
                'quality_mode': True,
                'description': 'Maximum Sharpness - Ultra MVS + 200K steps'
            }
        }

        # Get preset config or default to medium
        config = preset_configs.get(preset, preset_configs['medium'])
        config['quality_scale'] = quality_scale
        config['trainer'] = trainer

        # Allow advanced settings to override preset (skip None values to preserve preset defaults)
        if advanced_settings:
            config.update({k: v for k, v in advanced_settings.items() if v is not None})

        # Extract final settings
        detail_level = config['detail_level']
        training_steps = int(config['training_steps'] * quality_scale)
        enable_dense = config['enable_dense']
        max_image_size = config['max_image_size']
        quality_mode = config['quality_mode']

        # Initialize processing status first (so job-specific logs can be captured)
        processing_status[job_id] = {
            'status': 'processing',
            'step': 'Initializing...',
            'stage': 'initialization',
            'progress': 0,
            'error': None,
            'logs': deque(maxlen=200),
            'preset': preset,
            'config': config,
            'started_at': time.time(),
            'stages': {
                'feature_extraction': {'status': 'pending', 'started_at': None, 'elapsed': None},
                'feature_matching': {'status': 'pending', 'started_at': None, 'elapsed': None},
                'mapping': {'status': 'pending', 'started_at': None, 'elapsed': None},
                'dense_mvs': {'status': 'pending', 'started_at': None, 'elapsed': None},
                'training': {'status': 'pending', 'started_at': None, 'elapsed': None},
            }
        }

        # Assign a GPU to this job (for Brush and ML-Sharp)
        job_gpu = assign_gpu(job_id)
        add_log(f"GPU assigned for Brush/ML-Sharp: GPU {job_gpu} (of {GPU_COUNT} total)", "INFO")

        # Set current job for thread-local logging (must be after processing_status init)
        set_current_job(job_id)

        add_log(f"Starting job {job_id[:8]}...", "INFO")
        add_log(f"Preset: {preset.upper()} - {config.get('description', 'Custom settings')}", "INFO")
        add_log(f"Image path: {image_path}", "INFO")
        add_log(f"Matcher type: {matcher_type}", "INFO")
        add_log(f"Training steps: {training_steps:,}" if training_steps else f"Training steps: preset default", "INFO")
        if config.get('sharpness_boost'):
            add_log(f"[SHARPNESS] ULTRA SHARPNESS BOOST ENABLED - Maximum MVS quality!", "INFO")
        if quality_mode:
            add_log(f"[QUALITY] QUALITY MODE ENABLED - Maximum quality settings (slower processing)", "INFO")
        
        # Count images
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        add_log(f"Found {len(image_files)} images to process", "INFO")
        
        # Import and run the glomap processing
        from run_glomap import run_colmap, set_progress_callback
        
        # Set up progress callback to route COLMAP logs to our log system
        def colmap_progress_callback(message, level):
            add_log(message, level)
            # Detect stage transitions from COLMAP log messages
            if "STEP 2" in message or "Feature Matching" in message or "matching" in message.lower():
                transition_stage(processing_status, job_id, 'feature_matching')
            elif "STEP 3" in message or "Mapper" in message or "Reconstructing" in message or "triangulating" in message.lower():
                transition_stage(processing_status, job_id, 'mapping')
            elif "STEP 4" in message or "PatchMatch" in message or "MVS" in message or "dense" in message.lower():
                transition_stage(processing_status, job_id, 'dense_mvs')
            # Update status step with latest progress message
            if "Step" in message or "Processing" in message or "Matching" in message or "Registered" in message:
                processing_status[job_id]['step'] = message.replace("=", "").strip()

        set_progress_callback(colmap_progress_callback)
        
        # Check for COLMAP before starting - try multiple paths
        add_log("Checking COLMAP installation...", "INFO")

        # Add bundled COLMAP lib to PATH first (so DLLs are found without a system error popup)
        _app_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_app_dir)
        _bundled_colmap_bin_dir = os.path.join(_project_root, 'COLMAP', 'bin')
        _bundled_colmap_lib = os.path.join(_project_root, 'COLMAP', 'lib')
        _bundled_colmap_bin = os.path.join(_bundled_colmap_bin_dir, 'colmap.exe')
        _current_path = os.environ.get("PATH", "")
        if os.path.exists(_bundled_colmap_bin_dir) and _bundled_colmap_bin_dir not in _current_path:
            os.environ["PATH"] = _bundled_colmap_bin_dir + ";" + _current_path
        if os.path.exists(_bundled_colmap_lib) and _bundled_colmap_lib not in _current_path:
            os.environ["PATH"] = _bundled_colmap_lib + ";" + os.environ.get("PATH", "")

        # Try to find COLMAP - check PATH first, then common installation locations
        colmap_path = None
        possible_paths = [
            _bundled_colmap_bin,  # Bundled COLMAP (highest priority - correct version)
            "colmap",  # In PATH
            r"C:\COLMAP\bin\colmap.exe",
            r"C:\COLMAP\colmap.exe",
        ]

        # Also add system COLMAP bin/lib folders to PATH if present
        colmap_bin_dir = r"C:\COLMAP\bin"
        colmap_lib = r"C:\COLMAP\lib"
        _current_path = os.environ.get("PATH", "")
        if os.path.exists(colmap_bin_dir) and colmap_bin_dir not in _current_path:
            os.environ["PATH"] = colmap_bin_dir + ";" + _current_path
        if os.path.exists(colmap_lib) and colmap_lib not in _current_path:
            os.environ["PATH"] = colmap_lib + ";" + os.environ.get("PATH", "")

        # Add MSVC cl.exe to PATH for gsplat JIT CUDA kernel compilation
        _msvc_bin = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
        if os.path.exists(_msvc_bin):
            _msvc_versions = sorted(os.listdir(_msvc_bin), reverse=True)
            for _v in _msvc_versions:
                _cl_dir = os.path.join(_msvc_bin, _v, "bin", "Hostx64", "x64")
                if os.path.exists(os.path.join(_cl_dir, "cl.exe")):
                    if _cl_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = _cl_dir + ";" + os.environ.get("PATH", "")
                    add_log(f"MSVC cl.exe found at: {_cl_dir}", "INFO")
                    break

        for path in possible_paths:
            try:
                result = subprocess.run([path, "--help"], capture_output=True, timeout=10)
                if result.returncode == 0:
                    colmap_path = path
                    add_log(f"COLMAP found at: {path}", "INFO")
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if not colmap_path:
            add_log("COLMAP not found!", "ERROR")
            raise Exception("COLMAP is not installed or not in PATH. Please install COLMAP and add it to your PATH, or install it to C:\\COLMAP\\bin\\")

        # Check for COLMAP global_mapper (COLMAP 4.0+, replaces standalone GLOMAP)
        try:
            gm_check = subprocess.run([colmap_path, 'global_mapper', '--help'],
                                       capture_output=True, timeout=10)
            if gm_check.returncode == 0:
                add_log("COLMAP global_mapper available (COLMAP 4.0+ built-in GLOMAP)", "INFO")
            else:
                add_log("global_mapper not available - using COLMAP incremental mapper", "WARNING")
        except Exception:
            add_log("global_mapper not available - using COLMAP incremental mapper", "WARNING")

        processing_status[job_id]['step'] = 'Running COLMAP feature extraction...'
        transition_stage(processing_status, job_id, 'feature_extraction')
        processing_status[job_id]['progress'] = 20

        add_log("=" * 50, "INFO")
        add_log("STEP 1: Running COLMAP/GLOMAP Pipeline", "INFO")
        add_log("=" * 50, "INFO")
        add_log("Hardware: COLMAP auto-detects GPU/CPU (uses CUDA if available)", "INFO")

        # Pre-process images: resize large images and detect blurry ones
        max_image_size = config.get('max_image_size', 3840)
        if max_image_size and max_image_size > 0:
            resized = resize_images_for_colmap(image_path, max_size=max_image_size)
            if resized > 0:
                add_log(f"Resized {resized} images to max {max_image_size}px", "INFO")

        blurry = filter_blurry_images(image_path, blur_threshold=100.0)
        if blurry:
            add_log(f"Warning: {len(blurry)} potentially blurry images detected", "WARNING")
            for fname, score in blurry[:5]:
                add_log(f"  - {fname} (blur score: {score:.1f})", "WARNING")

        # Determine ultra_sharpness mode from config
        mvs_mode = config.get('mvs_quality_mode', 'balanced')
        ultra_sharpness_enabled = config.get('sharpness_boost', False) or (mvs_mode == 'ultra_sharpness')

        # Run COLMAP/GLOMAP processing
        # Pass enable_dense as override to run_colmap (preset's dense setting can be overridden by user)
        run_colmap(image_path, matcher_type, interval, '3dgs', detail_level, quality_mode, ultra_sharpness_enabled, enable_dense_override=enable_dense)
        
        add_log("COLMAP processing complete!", "INFO")
        processing_status[job_id]['step'] = 'COLMAP processing complete. Preparing for Gaussian Splat training...'
        transition_stage(processing_status, job_id, 'training')
        processing_status[job_id]['progress'] = 50
        
        # Check if sparse reconstruction exists
        parent_dir = os.path.dirname(image_path)
        sparse_path = os.path.join(parent_dir, 'sparse', '0')
        
        if not os.path.exists(sparse_path):
            add_log("Sparse reconstruction not found!", "ERROR")
            raise Exception("Sparse reconstruction not found. COLMAP processing may have failed.")
        
        add_log(f"Sparse reconstruction found at: {sparse_path}", "INFO")
        
        # Check how many images were actually registered
        try:
            import pycolmap
            reconstruction = pycolmap.Reconstruction(sparse_path)
            registered_count = len(reconstruction.images)
            points_count = len(reconstruction.points3D)
            add_log(f"COLMAP registered {registered_count} images with {points_count} 3D points", "INFO")
            
            # Warn if too few images were registered
            original_count = len([f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if registered_count < original_count * 0.5:
                add_log(f"WARNING: Only {registered_count}/{original_count} images were registered!", "WARNING")
                add_log("This means most images couldn't be matched. Try taking photos with more overlap.", "WARNING")
            
            # List registered images
            add_log("Registered images:", "DEBUG")
            for img_id, img in list(reconstruction.images.items())[:10]:
                add_log(f"  - {img.name}", "DEBUG")
            if registered_count > 10:
                add_log(f"  ... and {registered_count - 10} more", "DEBUG")
                
        except ImportError:
            add_log("pycolmap not available for detailed stats", "DEBUG")
        except Exception as e:
            add_log(f"Could not read reconstruction stats: {e}", "DEBUG")

        # Dense reconstruction (generates millions of points instead of thousands)
        # Check if dense reconstruction already ran inside run_colmap() (for presets with dense: True)
        # Presets that run dense in run_colmap: medium, high, ultra, extreme, maximum, insane, unlimited, dense, expert, sharpness
        dense_ply_path = None
        detail_levels_with_mvs = ['unlimited', 'dense', 'expert', 'medium', 'high', 'ultra', 'extreme', 'maximum', 'insane', 'sharpness']
        detail_level_includes_mvs = detail_level in detail_levels_with_mvs
        
        # Also check if dense PLY already exists (run_colmap may have already created it)
        potential_dense_ply = os.path.join(parent_dir, 'dense', 'fused.ply')
        dense_already_ran = os.path.exists(potential_dense_ply)
        
        if detail_level_includes_mvs:
            # MVS already ran (or is running) inside run_colmap() - DO NOT run dense_reconstruction.py again!
            # This prevents conflicts with existing dense/ workspace
            add_log("Dense reconstruction handled by COLMAP preset, checking for output...", "INFO")
            
            if os.path.exists(potential_dense_ply):
                dense_ply_path = potential_dense_ply
                add_log(f"Found dense PLY from COLMAP MVS: {dense_ply_path}", "INFO")
                # Copy to standard location
                dense_output = os.path.join(parent_dir, 'point_cloud_dense.ply')
                shutil.copy2(dense_ply_path, dense_output)
                add_log(f"Dense point cloud saved to: {dense_output}", "INFO")
            else:
                add_log("Dense reconstruction may still be processing, or MVS may have failed", "INFO")
                add_log("Check the logs for dense reconstruction progress", "INFO")
        elif dense_already_ran:
            # Dense PLY exists but preset doesn't include MVS - use existing output
            add_log("Found existing dense PLY from previous run, using it...", "INFO")
            dense_ply_path = potential_dense_ply
            dense_output = os.path.join(parent_dir, 'point_cloud_dense.ply')
            shutil.copy2(dense_ply_path, dense_output)
            add_log(f"Dense point cloud saved to: {dense_output}", "INFO")
        elif enable_dense:
            add_log("=" * 50, "INFO")
            add_log("STEP 2: Dense Reconstruction (High-Density Point Cloud)", "INFO")
            add_log("=" * 50, "INFO")

            processing_status[job_id]['step'] = 'Generating dense point cloud...'
            processing_status[job_id]['progress'] = 55

            try:
                from dense_reconstruction import run_dense_reconstruction

                add_log("Starting dense reconstruction for millions of points...", "INFO")
                add_log(f"Max image size: {max_image_size}px", "INFO")

                # Determine ultra_sharpness mode from config
                mvs_mode = config.get('mvs_quality_mode', 'balanced')
                ultra_sharpness = (mvs_mode == 'ultra_sharpness') or (detail_level == 'sharpness')
                
                dense_ply_path = run_dense_reconstruction(
                    parent_dir=parent_dir,
                    image_path=image_path,
                    sparse_path=sparse_path,
                    enable_dense=enable_dense,
                    max_image_size=max_image_size,
                    quality_mode=quality_mode,
                    ultra_sharpness_mode=ultra_sharpness,
                    colmap_path=colmap_path
                )

                if dense_ply_path and os.path.exists(dense_ply_path):
                    add_log(f"Dense reconstruction successful: {dense_ply_path}", "INFO")

                    # Copy dense PLY to main output directory
                    dense_output = os.path.join(parent_dir, 'point_cloud_dense.ply')
                    shutil.copy2(dense_ply_path, dense_output)
                    add_log(f"Dense point cloud saved to: {dense_output}", "INFO")
                else:
                    add_log("Dense reconstruction did not produce output, continuing with sparse only", "WARNING")

            except Exception as e:
                add_log(f"Dense reconstruction failed: {e}", "WARNING")
                add_log("Continuing with sparse reconstruction only", "INFO")
        else:
            add_log("Dense reconstruction disabled, using sparse points only", "INFO")

        add_log("=" * 50, "INFO")
        trainer_type = config.get('trainer', 'brush')
        if trainer_type == 'gsplat_mcmc':
            add_log("STEP 3: Training Gaussian Splats with gsplat MCMC", "INFO")
            processing_status[job_id]['step'] = 'Training with gsplat MCMC...'
        else:
            add_log("STEP 3: Training Gaussian Splats with Brush", "INFO")
            processing_status[job_id]['step'] = 'Training Gaussian Splats with Brush...'
        add_log("=" * 50, "INFO")
        processing_status[job_id]['progress'] = 60

        # Branch: gsplat MCMC trainer or Brush
        output_ply = None
        ply_generated = False

        if trainer_type == 'gsplat_mcmc':
            try:
                from gsplat_mcmc_trainer import train_mcmc
                add_log("Starting gsplat MCMC training...", "INFO")
                processing_status[job_id]['step'] = 'Training with gsplat MCMC...'

                def gsplat_progress(msg, level="INFO"):
                    add_log(f"gsplat: {msg}", level)
                    # Parse step number for progress
                    m = re.search(r'Step\s+(\d+)/(\d+)', str(msg))
                    if m:
                        step = int(m.group(1))
                        total = int(m.group(2))
                        pct = min(int(step / total * 90) + 60, 99)
                        processing_status[job_id]['progress'] = pct

                output_ply = os.path.join(parent_dir, 'gaussian_splat.ply')
                result = train_mcmc(
                    parent_dir,
                    total_steps=training_steps,
                    cap_max=1_500_000,
                    progress_callback=gsplat_progress,
                    output_ply_path=output_ply,
                )
                if result and os.path.exists(result):
                    ply_generated = True
                    add_log(f"gsplat MCMC training complete! Output: {result}", "INFO")
                else:
                    add_log("gsplat MCMC training failed — falling back to Brush", "WARNING")
                    trainer_type = 'brush'  # fall through to Brush
            except Exception as e:
                add_log(f"gsplat MCMC error: {e} — falling back to Brush", "WARNING")
                trainer_type = 'brush'  # fall through to Brush

        if trainer_type == 'brush' and not ply_generated:
            # Check if Brush is available - try multiple installation locations
            _app_dir_brush = os.path.dirname(os.path.abspath(__file__))
            _project_root_brush = os.path.dirname(_app_dir_brush)
            _bundled_brush = os.path.join(_project_root_brush, 'Brush', 'brush_app.exe')
            brush_paths = [
                _bundled_brush,           # Bundled Brush (highest priority)
                r"C:\Brush\brush_app.exe",
                r"C:\Brush\brush.exe",
                "brush_app",  # Check PATH
                "brush",      # Check PATH
            ]

            brush_path = None
            for path in brush_paths:
                if os.path.isabs(path) and os.path.exists(path):
                    brush_path = path
                    break
                else:
                    found = shutil.which(path)
                    if found:
                        brush_path = found
                        break

            brush_available = brush_path is not None
            if brush_available:
                add_log(f"Brush found at: {brush_path}", "INFO")
            else:
                add_log("Brush not found - will generate PLY from sparse reconstruction", "INFO")

            if brush_available:
                try:
                    images_folder = os.path.join(parent_dir, 'images')
                    source_folder = os.path.join(parent_dir, 'source')
                    input_folder = os.path.join(parent_dir, 'input')

                    def count_images(folder):
                        if os.path.exists(folder):
                            return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        return 0

                    images_count = count_images(images_folder)

                    if images_count > 0:
                        add_log(f"Using undistorted images folder ({images_count} images)", "INFO")
                    else:
                        source_count = count_images(source_folder)
                        input_count = count_images(input_folder)
                        add_log(f"No undistorted images found, checking source/input (source: {source_count}, input: {input_count})", "INFO")
                        if source_count > 0:
                            if os.path.exists(images_folder):
                                shutil.rmtree(images_folder)
                            shutil.copytree(source_folder, images_folder)
                            add_log(f"Created images folder from source ({source_count} images)", "INFO")
                        elif input_count > 0:
                            if os.path.exists(images_folder):
                                shutil.rmtree(images_folder)
                            shutil.copytree(input_folder, images_folder)
                            add_log(f"Created images folder from input ({input_count} images)", "INFO")

                    if os.path.exists(images_folder):
                        image_files = [f for f in os.listdir(images_folder)
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        add_log(f"Found {len(image_files)} images for Brush training", "INFO")
                    else:
                        add_log(f"Warning: images folder not found at {images_folder}", "WARNING")

                    # Run Brush training
                    add_log(f"Training with {training_steps} steps", "INFO")
                    export_path = parent_dir

                    sharpness_boost_enabled = config.get('sharpness_boost', False)
                    grad_threshold = "0.00002" if sharpness_boost_enabled else "0.00004"
                    growth_fraction = "0.15" if sharpness_boost_enabled else "0.1"
                    refine_every = "100" if sharpness_boost_enabled else "150"
                    max_res = "1920" if sharpness_boost_enabled else "1280"

                    brush_cmd = [
                        brush_path,
                        parent_dir,
                        "--total-steps", str(training_steps),
                        "--export-path", export_path,
                        "--export-name", "gaussian_splat.ply",
                        "--export-every", str(training_steps),
                        "--max-resolution", max_res,
                        "--growth-grad-threshold", grad_threshold,
                        "--growth-select-fraction", growth_fraction,
                        "--refine-every", refine_every,
                    ]

                    add_log(f"Quality settings: grad_threshold={grad_threshold}", "INFO")
                    add_log(f"Starting Brush training with {training_steps} steps...", "INFO")
                    add_log(f"Brush command: {' '.join(brush_cmd)}", "DEBUG")
                    processing_status[job_id]['step'] = f'Training Gaussian Splats (0/{training_steps} steps)...'

                    timeout_seconds = max(7200, (training_steps // 500) * 60 + 3600)
                    add_log(f"Brush pinned to GPU {job_gpu} via CUDA_VISIBLE_DEVICES", "INFO")
                    process = subprocess.Popen(
                        brush_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=gpu_env(job_gpu)
                    )

                    # Brush produces no stdout — estimate progress by elapsed time
                    training_start = time.time()
                    while process.poll() is None:
                        elapsed = time.time() - training_start
                        # Estimate: ~0.5s per step on average (adjusts as it runs)
                        est_total_time = max(training_steps * 0.5, 60)
                        pct = min(99, int(elapsed / est_total_time * 100))
                        est_step = min(training_steps - 1, int(elapsed / 0.5))
                        processing_status[job_id]['step'] = f'Training Gaussian Splats (~{est_step}/{training_steps} steps, {int(elapsed)}s elapsed)...'
                        processing_status[job_id]['progress'] = pct
                        if elapsed > timeout_seconds:
                            process.kill()
                            add_log("Brush timed out", "ERROR")
                            break
                        time.sleep(5)

                    result_code = process.returncode
                    add_log(f"Brush completed with return code: {result_code}", "INFO")

                    if result_code == 0:
                        possible_plys = [
                            os.path.join(export_path, f"export_{training_steps}.ply"),
                            os.path.join(export_path, "gaussian_splat.ply"),
                            os.path.join(export_path, "splat.ply"),
                        ]
                        export_plys = list(Path(export_path).glob('export_*.ply'))
                        if export_plys:
                            possible_plys.extend([str(p) for p in export_plys])

                        found_ply = None
                        for ply_path in possible_plys:
                            if os.path.exists(ply_path):
                                found_ply = ply_path
                                break

                        if found_ply:
                            output_ply = os.path.join(parent_dir, 'gaussian_splat.ply')
                            if found_ply != output_ply:
                                shutil.copy(found_ply, output_ply)
                            ply_generated = True
                            ply_size = os.path.getsize(output_ply) / (1024 * 1024)
                            add_log(f"Brush training complete! Output: {output_ply} ({ply_size:.2f} MB)", "INFO")
                        else:
                            add_log(f"Brush completed but no PLY found in {export_path}", "WARNING")
                            all_plys = list(Path(export_path).glob('*.ply'))
                            if all_plys:
                                output_ply = str(all_plys[0])
                                ply_generated = True
                                add_log(f"Found PLY file: {output_ply}", "INFO")
                    else:
                        add_log(f"Brush training failed with code {result_code}", "ERROR")

                except Exception as e:
                    add_log(f"Error running Brush: {e}", "ERROR")

        # Fallback: Generate basic point cloud from COLMAP if Brush failed
        if not ply_generated:
            processing_status[job_id]['step'] = 'Generating point cloud from COLMAP...'
            processing_status[job_id]['progress'] = 80
            
            try:
                from gaussian_splat_utils import generate_ply_from_colmap
                output_ply = os.path.join(parent_dir, 'point_cloud.ply')
                ply_generated = generate_ply_from_colmap(sparse_path, output_ply)
                if ply_generated:
                    print(f"Generated basic point cloud: {output_ply}")
            except Exception as e:
                print(f"Error generating point cloud: {e}")
                ply_generated = False
        
        add_log("=" * 50, "INFO")
        add_log("PROCESSING COMPLETE!", "INFO")
        add_log("=" * 50, "INFO")
        add_log(f"PLY generated: {ply_generated}", "INFO")
        if output_ply:
            add_log(f"Output file: {output_ply}", "INFO")
        
        processing_status[job_id]['step'] = 'Processing complete!'
        processing_status[job_id]['progress'] = 100
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_path'] = parent_dir
        processing_status[job_id]['sparse_path'] = sparse_path
        processing_status[job_id]['ply_path'] = output_ply if ply_generated else None
        processing_status[job_id]['ply_available'] = ply_generated
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error messages for common COLMAP failures
        if "No good initial image pair found" in error_msg or "failed to create sparse model" in error_msg:
            error_msg = """COLMAP could not reconstruct your scene. This usually means:

1. Images don't have enough overlap - take photos with 60-80% overlap between consecutive images
2. Images are of different scenes - all images should be of the same subject
3. Not enough features - avoid plain/textureless surfaces
4. Images are blurry - use sharp, well-lit photos
5. Too few images - try using at least 10-20 images

Tips:
- Walk around the object slowly, taking photos every few steps
- Keep the subject centered in all photos
- Ensure good lighting with minimal shadows
- Avoid reflective or transparent surfaces"""
        
        add_log(f"Processing failed: {error_msg[:200]}", "ERROR")
        if job_id in processing_status:
            processing_status[job_id]['status'] = 'error'
            processing_status[job_id]['error'] = error_msg
        else:
            processing_status[job_id] = {
                'status': 'error',
                'error': error_msg,
                'progress': 0
            }
    finally:
        release_gpu(job_id)

def process_mlsharp_async(job_id, image_path, render_views=False, device='auto'):
    """Process single image using ml-sharp for ultra-fast 3D Gaussian splat generation"""
    try:
        set_current_job(job_id)
        processing_status[job_id] = {
            'status': 'processing',
            'step': 'Initializing ML-Sharp...',
            'progress': 0,
            'error': None,
            'logs': deque(maxlen=200),
            'method': 'mlsharp',
            'config': {'render_views': render_views, 'device': device}
        }

        # Assign GPU for this job
        job_gpu = assign_gpu(job_id)
        add_log(f"ML-Sharp pinned to GPU {job_gpu} via CUDA_VISIBLE_DEVICES", "INFO")

        add_log(f"Starting ML-Sharp job {job_id[:8]}...", "INFO")
        add_log(f"Device: {device}, Render views: {render_views}", "INFO")

        # Check ML-Sharp availability
        if not mlsharp_available:
            raise Exception(
                "ML-Sharp is not installed. Install from source:\n"
                "  git clone https://github.com/apple/ml-sharp.git\n"
                "  cd ml-sharp\n"
                "  pip install -r requirements.txt\n"
                "  sharp --version"
            )

        add_log(f"ML-Sharp version: {mlsharp_version}", "INFO")

        # Find input image
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            raise Exception("No valid image file found in upload")
        if len(image_files) > 1:
            add_log(f"Warning: Multiple images found, using {image_files[0]}", "WARNING")

        input_image = os.path.join(image_path, image_files[0])
        parent_dir = os.path.dirname(image_path)
        output_dir = os.path.join(parent_dir, 'mlsharp_output')
        os.makedirs(output_dir, exist_ok=True)

        add_log(f"Input image: {input_image}", "INFO")
        add_log(f"Output directory: {output_dir}", "INFO")

        # Check image size (if PIL available)
        try:
            from PIL import Image
            img = Image.open(input_image)
            width, height = img.size
            add_log(f"Image size: {width}x{height}", "INFO")
            if width < 256 or height < 256:
                add_log("Warning: Small image may produce low-quality results", "WARNING")
            if width > 4096 or height > 4096:
                add_log("Warning: Large image may cause memory issues", "WARNING")
        except ImportError:
            add_log("PIL not available, skipping image validation", "DEBUG")

        # Update progress
        processing_status[job_id]['step'] = 'Loading ML-Sharp model...'
        processing_status[job_id]['progress'] = 20

        # Build ml-sharp command — always use cuda, pin to assigned GPU
        mlsharp_cmd = ['sharp', 'predict', '-i', input_image, '-o', output_dir]
        if render_views:
            mlsharp_cmd.append('--render')
            add_log("Novel view rendering enabled", "INFO")
        effective_device = device if device not in ('auto', 'cuda') else 'cuda'
        mlsharp_cmd.extend(['--device', effective_device])
        add_log(f"ML-Sharp device: {effective_device} (GPU {job_gpu})", "INFO")

        add_log(f"Command: {' '.join(mlsharp_cmd)}", "DEBUG")

        # Run ml-sharp with progress tracking
        processing_status[job_id]['step'] = 'Running ML-Sharp prediction...'
        processing_status[job_id]['progress'] = 40

        start_time = time.time()

        mlsharp_env = gpu_env(job_gpu)
        mlsharp_env['PYTHONHTTPSVERIFY'] = '0'
        mlsharp_env['CURL_CA_BUNDLE'] = ''

        process = subprocess.Popen(
            mlsharp_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=mlsharp_env
        )

        # Stream output and track progress
        model_loaded = False
        predicting = False

        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            add_log(f"ML-Sharp: {line}", "DEBUG")

            # Parse progress indicators
            line_lower = line.lower()

            if 'loading' in line_lower and 'model' in line_lower:
                if not model_loaded:
                    processing_status[job_id]['step'] = 'Loading model checkpoint...'
                    processing_status[job_id]['progress'] = 50
                    add_log("Loading model checkpoint...", "INFO")
                    model_loaded = True
            elif 'download' in line_lower:
                processing_status[job_id]['step'] = 'Downloading model (first run)...'
                processing_status[job_id]['progress'] = 30
                add_log("Downloading model checkpoint (~500MB, first run only)", "INFO")
            elif any(word in line_lower for word in ['predict', 'inference', 'forward']):
                if not predicting:
                    processing_status[job_id]['step'] = 'Predicting 3D Gaussians...'
                    processing_status[job_id]['progress'] = 70
                    add_log("Generating 3D Gaussian splat...", "INFO")
                    predicting = True
            elif 'saving' in line_lower or 'writing' in line_lower or 'export' in line_lower:
                processing_status[job_id]['step'] = 'Saving PLY file...'
                processing_status[job_id]['progress'] = 90
                add_log("Writing output PLY...", "INFO")
            elif 'error' in line_lower or 'fail' in line_lower:
                add_log(f"Error: {line}", "ERROR")

        # Wait for completion
        return_code = process.wait()
        elapsed = time.time() - start_time

        if return_code != 0:
            raise Exception(f"ML-Sharp failed with exit code {return_code}")

        add_log(f"ML-Sharp completed in {elapsed:.2f} seconds", "INFO")

        # Find output PLY file
        ply_files = list(Path(output_dir).glob('*.ply'))
        if not ply_files:
            raise Exception("ML-Sharp did not produce PLY output")

        output_ply = str(ply_files[0])
        ply_size = os.path.getsize(output_ply) / (1024 * 1024)
        add_log(f"Output PLY: {output_ply} ({ply_size:.2f} MB)", "INFO")

        # Copy to standard location for compatibility with viewer
        final_ply = os.path.join(parent_dir, 'gaussian_splat.ply')
        shutil.copy2(output_ply, final_ply)
        add_log(f"Copied to: {final_ply}", "INFO")

        # Mark complete
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['step'] = 'Processing complete!'
        processing_status[job_id]['progress'] = 100
        processing_status[job_id]['output_path'] = parent_dir
        processing_status[job_id]['ply_path'] = final_ply
        processing_status[job_id]['ply_available'] = True
        processing_status[job_id]['elapsed_time'] = elapsed

        add_log("=" * 50, "INFO")
        add_log("ML-SHARP PROCESSING COMPLETE!", "INFO")
        add_log(f"Total time: {elapsed:.2f} seconds", "INFO")
        add_log("=" * 50, "INFO")

    except Exception as e:
        error_msg = str(e)

        # Provide helpful error messages
        if "not installed" in error_msg.lower():
            error_msg = """ML-Sharp is not installed. To enable:

1. Clone and install from source:
   git clone https://github.com/apple/ml-sharp.git
   cd ml-sharp
   pip install -r requirements.txt

2. Verify: sharp --version

3. Restart the server

For more info: https://github.com/apple/ml-sharp"""
        elif "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            error_msg = """Out of memory! Try:

1. Use CPU instead of GPU (slower but uses less memory)
2. Use a smaller input image
3. Close other applications
4. Upgrade GPU memory if possible"""
        elif "cuda" in error_msg.lower() and "not available" in error_msg.lower():
            error_msg = """CUDA not available. ML-Sharp will use CPU.

For GPU acceleration:
1. Install CUDA toolkit
2. Install PyTorch with CUDA support
3. Restart server

Continuing with CPU (slower)..."""

        add_log(f"ML-Sharp failed: {error_msg}", "ERROR")
        if job_id in processing_status:
            processing_status[job_id]['status'] = 'error'
            processing_status[job_id]['error'] = error_msg
        else:
            processing_status[job_id] = {
                'status': 'error',
                'error': error_msg,
                'progress': 0
            }
    finally:
        release_gpu(job_id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gpu-info')
def gpu_info():
    """Get GPU information for time estimation"""
    try:
        # Try to get NVIDIA GPU info
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)

        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split(',')[0].strip()

            # Categorize GPU speed (multiplier for time estimates)
            # Higher-end GPUs = lower multiplier (faster)
            speed_multiplier = 1.0  # Default

            gpu_lower = gpu_name.lower()
            if 'rtx 40' in gpu_lower or 'rtx 50' in gpu_lower:
                speed_multiplier = 0.6  # RTX 40/50 series - very fast
            elif 'rtx 30' in gpu_lower:
                speed_multiplier = 0.8  # RTX 30 series - fast
            elif 'rtx 20' in gpu_lower or 'gtx 16' in gpu_lower:
                speed_multiplier = 1.0  # RTX 20 / GTX 16 series - medium
            elif 'gtx 10' in gpu_lower or 'rtx' in gpu_lower:
                speed_multiplier = 1.2  # Older cards - slower
            else:
                speed_multiplier = 1.5  # Unknown/older GPU

            return jsonify({
                'has_gpu': True,
                'gpu_name': gpu_name,
                'speed_multiplier': speed_multiplier
            })
        else:
            return jsonify({
                'has_gpu': False,
                'speed_multiplier': 2.0  # CPU fallback - much slower
            })
    except Exception as e:
        return jsonify({
            'has_gpu': False,
            'speed_multiplier': 1.0,
            'error': str(e)
        })

@app.route('/mlsharp-info')
def mlsharp_info():
    """Get ML-Sharp installation status and info"""
    return jsonify({
        'available': mlsharp_available,
        'version': mlsharp_version,
        'install_command': 'git clone https://github.com/apple/ml-sharp.git && cd ml-sharp && pip install -r requirements.txt',
        'docs_url': 'https://github.com/apple/ml-sharp'
    })

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up all old processing jobs"""
    try:
        cleanup_all_jobs()
        return jsonify({'message': 'All processing jobs cleaned up successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup-old', methods=['POST'])
def cleanup_old():
    """Clean up processing jobs older than 24 hours"""
    try:
        cleanup_old_jobs(max_age_hours=24)
        return jsonify({'message': 'Old processing jobs cleaned up successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/kill', methods=['POST'])
def kill_all_processes():
    """Kill all background processes (COLMAP, GLOMAP, Brush, etc.)"""
    killed = []
    process_names = ['colmap', 'glomap', 'brush', 'brush_app']
    
    try:
        import psutil
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_name = proc.info['name'].lower() if proc.info['name'] else ''
                proc_cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                
                # Check if it's one of our processes
                for target in process_names:
                    if target in proc_name or target in proc_cmdline:
                        proc.kill()
                        killed.append(f"{proc.info['name']} (PID: {proc.info['pid']})")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
            except Exception:
                pass
        
        # Clear processing status
        for job_id in list(processing_status.keys()):
            if processing_status[job_id].get('status') == 'processing':
                processing_status[job_id]['status'] = 'cancelled'
                processing_status[job_id]['error'] = 'Cancelled by user'
        
        add_log(f"Killed processes: {killed if killed else 'None running'}", "WARNING")
        
        return jsonify({
            'message': f'Killed {len(killed)} processes. {", ".join(killed) if killed else "No target processes were running."}',
            'killed': killed
        })
    except ImportError:
        # psutil not installed, try using subprocess
        try:
            for name in process_names:
                subprocess.run(['taskkill', '/F', '/IM', f'{name}.exe'], capture_output=True)
                subprocess.run(['taskkill', '/F', '/IM', f'{name}'], capture_output=True)
            
            # Clear processing status
            for job_id in list(processing_status.keys()):
                if processing_status[job_id].get('status') == 'processing':
                    processing_status[job_id]['status'] = 'cancelled'
                    processing_status[job_id]['error'] = 'Cancelled by user'
            
            add_log("Killed processes using taskkill", "WARNING")
            return jsonify({'message': 'Killed processes using taskkill (psutil not available)'})
        except Exception as e:
            return jsonify({'error': f'Failed to kill processes: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-for-view', methods=['POST'])
def upload_for_view():
    """Upload a .ply file directly for viewing in the browser without processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'Empty file'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ('ply', 'splat'):
        return jsonify({'error': 'Only .ply and .splat files are supported'}), 400

    token = str(uuid.uuid4())
    job_folder = os.path.join(app.config['PROCESSING_FOLDER'], token)
    os.makedirs(job_folder, exist_ok=True)

    filename = secure_filename(file.filename)
    file.save(os.path.join(job_folder, filename))
    add_log(f"Uploaded splat for viewing: {filename} -> {token[:8]}", "INFO")

    return jsonify({'token': token})

@app.route('/logs')
def logs_page():
    """Show the live log viewer page"""
    return render_template('logs.html')

@app.route('/logs/stream')
def logs_stream():
    """Stream logs using Server-Sent Events"""
    def generate():
        last_index = 0
        while True:
            current_logs = get_logs()
            if len(current_logs) > last_index:
                for log in current_logs[last_index:]:
                    yield f"data: {log}\n\n"
                last_index = len(current_logs)
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/logs/current')
def logs_current():
    """Get current logs as JSON (wrapped)"""
    return jsonify({'logs': get_logs()})

@app.route('/logs/json')
def logs_json():
    """Get current logs as JSON array (raw)"""
    return jsonify(get_logs())

@app.route('/logs/json/<job_id>')
def logs_json_job(job_id):
    """Get logs for a specific job as JSON array"""
    return jsonify(get_job_logs(job_id))

@app.route('/logs/clear', methods=['POST'])
def logs_clear():
    """Clear the log buffer"""
    with log_lock:
        log_buffer.clear()
    add_log("Logs cleared", "INFO")
    return jsonify({'message': 'Logs cleared'})

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle image upload"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    # Get processing method
    method = request.form.get('method', 'traditional')

    # Validate method
    if method not in ['traditional', 'mlsharp']:
        return jsonify({'error': 'Invalid processing method'}), 400

    # Check ML-Sharp availability if requested
    if method == 'mlsharp' and not mlsharp_available:
        return jsonify({
            'error': 'ML-Sharp is not installed. Install from source: https://github.com/apple/ml-sharp'
        }), 400

    # Check concurrent job limit (only for traditional pipeline)
    if method == 'traditional':
        active_jobs = get_active_jobs_count()
        if active_jobs >= MAX_CONCURRENT_JOBS:
            return jsonify({
                'error': f'Too many jobs running. Please wait for current jobs to complete. (Limit: {MAX_CONCURRENT_JOBS} concurrent jobs)',
                'active_jobs': active_jobs
            }), 429

    # Create unique job ID
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(app.config['PROCESSING_FOLDER'], job_id)
    os.makedirs(job_folder, exist_ok=True)

    # Track job creation time
    job_start_times[job_id] = time.time()
    
    images_folder = os.path.join(job_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)
    
    # Handle uploaded files - check all files, not just the first one
    # Categorize files by type to prevent ignoring files when first file is video/ZIP
    zip_files = [f for f in files if f.filename.endswith('.zip')]
    video_files = [f for f in files if is_video_file(f.filename)]
    image_files = [f for f in files if f and allowed_file(f.filename) and not f.filename.endswith('.zip') and not is_video_file(f.filename)]

    # ML-Sharp specific validation
    if method == 'mlsharp':
        if len(zip_files) > 0 or len(video_files) > 0:
            return jsonify({'error': 'ML-Sharp only accepts image files (no video/ZIP)'}), 400
        if len(image_files) != 1:
            return jsonify({'error': 'ML-Sharp requires exactly one image'}), 400

    # Only allow one type of upload at a time to avoid confusion
    if len(zip_files) > 0 and (len(video_files) > 0 or len(image_files) > 0):
        return jsonify({'error': 'Cannot mix ZIP files with other files. Please upload only a ZIP, only a video, or only images.'}), 400
    if len(video_files) > 0 and len(image_files) > 0:
        return jsonify({'error': 'Cannot mix video files with image files. Please upload only a video or only images.'}), 400
    if len(video_files) > 1:
        return jsonify({'error': 'Please upload only one video file at a time.'}), 400
    if len(zip_files) > 1:
        return jsonify({'error': 'Please upload only one ZIP file at a time.'}), 400
    
    is_video = False  # Default
    
    if len(zip_files) == 1:
        # ZIP file
        uploaded_file = zip_files[0]
        zip_path = os.path.join(job_folder, secure_filename(uploaded_file.filename))
        uploaded_file.save(zip_path)
        extract_zip(zip_path, images_folder)
        add_log(f"Extracted ZIP file: {uploaded_file.filename}", "INFO")
        
    elif len(video_files) == 1:
        # Video file - extract frames
        uploaded_file = video_files[0]
        video_path = os.path.join(job_folder, secure_filename(uploaded_file.filename))
        uploaded_file.save(video_path)
        add_log(f"Uploaded video: {uploaded_file.filename}", "INFO")
        
        # Get frame interval from form (default to 10 for video)
        frame_interval = int(request.form.get('interval', 10))
        # Allow any interval, including 1 (every frame)
        
        # Get max frames from form (default to 1000 for better coverage)
        max_frames = int(request.form.get('max_frames', 1000))
        
        # Extract frames
        frame_count = extract_frames_from_video(video_path, images_folder, frame_interval=frame_interval, max_frames=max_frames)
        
        if frame_count == 0:
            return jsonify({'error': 'Could not extract frames from video'}), 400
        
        add_log(f"Extracted {frame_count} frames from video", "INFO")
        
        # Mark that interval was already applied during extraction
        # So COLMAP should NOT apply interval again
        is_video = True
        
    elif len(image_files) > 0:
        # Save all individual image files
        saved_count = 0
        for file in image_files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(images_folder, filename))
            saved_count += 1
        add_log(f"Uploaded {saved_count} image files", "INFO")
    else:
        return jsonify({'error': 'No valid files uploaded. Please upload images, a video, or a ZIP file.'}), 400
    
    # Get processing parameters - NEW SIMPLIFIED PRESET SYSTEM
    preset = request.form.get('preset', 'medium')  # low, medium, or high
    matcher_type = request.form.get('matcher_type', 'exhaustive_matcher')
    interval = int(request.form.get('interval', 1))

    # Auto-detect matcher type if set to "auto"
    if matcher_type == 'auto':
        if is_video:
            matcher_type = 'sequential_matcher'
            add_log("Auto-detected matcher: Sequential (video input)", "INFO")
        else:
            matcher_type = 'exhaustive_matcher'
            add_log("Auto-detected matcher: Exhaustive (image input)", "INFO")

    # For videos, interval was already applied during frame extraction
    # So don't apply it again in COLMAP
    if is_video:
        colmap_interval = 1  # Don't filter again
        add_log(f"Video mode: interval={interval} already applied during extraction", "DEBUG")
    else:
        colmap_interval = interval  # Apply interval for uploaded images

    # Handle Sharpness & Training options (work with ANY preset)
    sharpness_boost = request.form.get('sharpness_boost', 'false').lower() == 'true'

    # Quality scale multiplier: draft=0.3x, standard=1x, cinematic=2x
    quality_scale_name = request.form.get('quality_scale', 'standard')
    quality_scale_map = {'draft': 0.3, 'standard': 1.0, 'cinematic': 2.0}
    quality_scale = quality_scale_map.get(quality_scale_name, 1.0)

    # Trainer type: brush (default) or gsplat_mcmc
    trainer_type = request.form.get('trainer', 'brush')

    # Build advanced settings from form data (always available, behind details toggle)
    mvs_mode = request.form.get('mvs_quality_mode', 'balanced').lower()
    quality_mode_enabled = mvs_mode in ['quality', 'ultra_sharpness']

    advanced_settings = {
        'training_steps': request.form.get('training_steps', None),
        'enable_dense': request.form.get('enable_dense', 'true').lower() == 'true',
        'max_image_size': int(request.form.get('max_image_size', 3200)),
        'quality_mode': quality_mode_enabled,
        'mvs_quality_mode': mvs_mode,
        'trainer': trainer_type,
    }
    # Convert training_steps to int if provided
    if advanced_settings['training_steps'] is not None:
        advanced_settings['training_steps'] = int(advanced_settings['training_steps'])

    # Apply sharpness boost
    if sharpness_boost:
        advanced_settings['sharpness_boost'] = True
        advanced_settings['mvs_quality_mode'] = 'ultra_sharpness'
        advanced_settings['quality_mode'] = True
        add_log("[SHARPNESS] Ultra Sharpness MVS boost enabled!", "INFO")

    # Route to appropriate processor based on method
    if method == 'mlsharp':
        # ML-Sharp processing
        render_views = request.form.get('mlsharp_render', 'false').lower() == 'true'
        device = request.form.get('mlsharp_device', 'auto')

        add_log(f"Starting ML-Sharp job with device={device}, render={render_views}", "INFO")

        thread = threading.Thread(
            target=process_mlsharp_async,
            args=(job_id, images_folder, render_views, device)
        )
    else:
        # Traditional COLMAP pipeline
        # Start async processing with new preset system
        thread = threading.Thread(
            target=process_images_async,
            args=(job_id, images_folder, preset, matcher_type, colmap_interval, advanced_settings, quality_scale, trainer_type)
        )

    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': 'Upload successful. Processing started.',
        'method': method
    })

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    # Convert to JSON-serializable dict (deque can't be serialized directly)
    status = processing_status[job_id].copy()
    if 'logs' in status:
        status['logs'] = list(status['logs'])  # Convert deque to list
    
    return jsonify(status)

@app.route('/ply/<job_id>.ply')
@app.route('/ply/<job_id>')
def serve_ply(job_id):
    """Serve PLY file for viewer"""
    # First try from processing_status (current session)
    if job_id in processing_status:
        job_data = processing_status[job_id]
        if job_data['status'] != 'completed':
            return jsonify({'error': 'Processing not complete'}), 400
        
        output_path = job_data.get('output_path')
        if output_path:
            ply_path = job_data.get('ply_path')
            if ply_path and os.path.exists(ply_path):
                return send_file(ply_path, mimetype='application/octet-stream')
            
            ply_files = list(Path(output_path).rglob('*.ply'))
            if ply_files:
                return send_file(str(ply_files[0]), mimetype='application/octet-stream')
    
    # Fallback: check filesystem directly for jobs from previous sessions
    job_folder = os.path.join(app.config['PROCESSING_FOLDER'], job_id)
    if os.path.exists(job_folder):
        ply_files = list(Path(job_folder).rglob('*.ply'))
        if ply_files:
            return send_file(str(ply_files[0]), mimetype='application/octet-stream')
        
        # Try to generate PLY if sparse reconstruction exists
        sparse_path = os.path.join(job_folder, 'sparse', '0')
        if os.path.exists(sparse_path):
            output_ply = os.path.join(job_folder, 'point_cloud.ply')
            try:
                from gaussian_splat_utils import generate_ply_from_colmap
                if generate_ply_from_colmap(sparse_path, output_ply):
                    return send_file(output_ply, mimetype='application/octet-stream')
            except Exception as e:
                print(f"Error generating PLY: {e}")
    
    return jsonify({'error': 'PLY file not found'}), 404

@app.route('/download/<job_id>/<file_type>')
def download_file(job_id, file_type):
    """Download processed files"""
    # Get output_path from processing_status or filesystem
    output_path = None
    sparse_path = None
    
    if job_id in processing_status:
        job_data = processing_status[job_id]
        if job_data['status'] != 'completed':
            return jsonify({'error': 'Processing not complete'}), 400
        output_path = job_data.get('output_path')
        sparse_path = job_data.get('sparse_path')
    else:
        # Fallback to filesystem for jobs from previous sessions
        job_folder = os.path.join(app.config['PROCESSING_FOLDER'], job_id)
        if os.path.exists(job_folder):
            output_path = job_folder
            sparse_path = os.path.join(job_folder, 'sparse', '0')
    
    if not output_path:
        return jsonify({'error': 'Job not found'}), 404
    
    if file_type == 'ply':
        # Use custom filename if provided via query param, strip unsafe chars
        raw_name = request.args.get('name', '').strip()
        safe_name = ''.join(c for c in raw_name if c.isalnum() or c in ('_', '-')) or 'gaussian_splat'
        download_name = safe_name + '.ply'

        # Look for .ply file in output directory
        ply_files = list(Path(output_path).rglob('*.ply'))
        if ply_files:
            return send_file(str(ply_files[0]), as_attachment=True, download_name=download_name)

        # Try to generate PLY if sparse reconstruction exists
        if sparse_path and os.path.exists(sparse_path):
            output_ply = os.path.join(output_path, 'point_cloud.ply')
            try:
                from gaussian_splat_utils import generate_ply_from_colmap
                if generate_ply_from_colmap(sparse_path, output_ply):
                    return send_file(output_ply, as_attachment=True, download_name=download_name)
            except Exception as e:
                print(f"Error generating PLY: {e}")
        
        # Return a helpful message
        return jsonify({
            'error': 'PLY file not found. A basic point cloud PLY was not generated.',
            'message': 'For full Gaussian Splat .ply files, training is required.',
            'note': 'You can download the sparse reconstruction and use it with 3D Gaussian Splatting training or Brush.'
        }), 404
    elif file_type == 'sparse':
        # Zip the sparse reconstruction
        if sparse_path and os.path.exists(sparse_path):
            import io
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(sparse_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, sparse_path)
                        zipf.write(file_path, arcname)
            zip_buffer.seek(0)
            return send_file(zip_buffer, as_attachment=True, download_name='sparse_reconstruction.zip', mimetype='application/zip')
        else:
            return jsonify({'error': 'Sparse reconstruction not found'}), 404
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/view/<job_id>')
def view_result(job_id):
    """Redirect to SuperSplat viewer"""
    return redirect(f'/static/supersplat/index.html?load=/ply/{job_id}.ply')

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': f'Internal server error: {str(error)}'}), 500

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({'error': 'File too large. Maximum upload size is 2GB.'}), 413

if __name__ == '__main__':
    # Check ML-Sharp availability on startup
    check_mlsharp_availability()

    # Start background cleanup thread
    start_background_cleanup()
    add_log("Gaussian Splatting Server starting...", "INFO")
    add_log(f"Max concurrent jobs: {MAX_CONCURRENT_JOBS}", "INFO")
    add_log(f"Auto-cleanup interval: {AUTO_CLEANUP_HOURS} hours", "INFO")
    add_log(f"ML-Sharp available: {mlsharp_available}", "INFO")
    if mlsharp_available:
        add_log(f"ML-Sharp version: {mlsharp_version}", "INFO")

    app.run(debug=True, host='0.0.0.0', port=5000, use_debugger=False, use_reloader=False)

