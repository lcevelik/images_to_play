#!/usr/bin/env python3
"""
Test script for images_to_play presets.
Uploads images and runs low, medium, high presets.
"""
import requests
import time
import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:5000"
SOURCE_DIR = r"F:\Codebase\images_to_play\simple_splat\App\processing\8e12c030-af50-447e-a31c-f3d0cc856b2c\source"

def get_image_files():
    """Get list of image files from source directory."""
    files = []
    for f in os.listdir(SOURCE_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            files.append(os.path.join(SOURCE_DIR, f))
    return sorted(files)

def upload_and_process(preset, files, max_files=20):
    """Upload files and start processing with given preset."""
    print(f"\n{'='*60}")
    print(f"Testing preset: {preset.upper()}")
    print(f"{'='*60}")
    
    # Use subset of files for faster testing
    test_files = files[:max_files]
    print(f"Using {len(test_files)} images")
    
    # Prepare files for upload
    upload_files = []
    for fpath in test_files:
        fname = os.path.basename(fpath)
        upload_files.append(('files', (fname, open(fpath, 'rb'), 'image/jpeg')))
    
    # Upload with preset
    data = {
        'method': 'traditional',
        'preset': preset,
        'matcher_type': 'exhaustive_matcher',
        'interval': '1',
        'quality_scale': 'standard',
        'trainer': 'brush',
        'enable_dense': 'true' if preset != 'low' else 'false',
        'max_image_size': '3200',
        'training_steps': '10000',
    }
    
    try:
        response = requests.post(f"{BASE_URL}/upload", files=upload_files, data=data)
        result = response.json()
        
        if response.status_code == 200:
            job_id = result.get('job_id')
            print(f"✓ Job started: {job_id}")
            return job_id
        else:
            print(f"✗ Upload failed: {result.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    finally:
        # Close file handles
        for _, (_, fh, _) in upload_files:
            fh.close()

def monitor_job(job_id, preset, timeout=600):
    """Monitor job until completion or timeout."""
    print(f"\nMonitoring job {job_id}...")
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/status/{job_id}")
            status = response.json()
            
            current_status = status.get('status', 'unknown')
            progress = status.get('progress', 0)
            stage = status.get('stage', '')
            
            if current_status != last_status:
                print(f"  Status: {current_status} | Progress: {progress:.1f}% | Stage: {stage}")
                last_status = current_status
            
            if current_status in ['complete', 'error', 'failed']:
                return status
            
            time.sleep(2)
        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(5)
    
    print(f"  Timeout after {timeout}s")
    return None

def check_output(job_id, preset):
    """Check the output PLY file."""
    ply_path = f"F:\\Codebase\\images_to_play\\simple_splat\\App\\processing\\{job_id}\\gaussian_splat.ply"
    sparse_path = f"F:\\Codebase\\images_to_play\\simple_splat\\App\\processing\\{job_id}\\sparse\\0"
    dense_path = f"F:\\Codebase\\images_to_play\\simple_splat\\App\\processing\\{job_id}\\dense\\fused.ply"
    
    print(f"\nOutput for {preset.upper()}:")
    
    # Check if PLY exists
    if os.path.exists(ply_path):
        size = os.path.getsize(ply_path)
        print(f"  ✓ gaussian_splat.ply: {size/1024/1024:.2f} MB")
    else:
        print(f"  ✗ gaussian_splat.ply: Not found")
    
    # Check sparse reconstruction
    if os.path.exists(sparse_path):
        files = os.listdir(sparse_path)
        print(f"  ✓ Sparse reconstruction: {len(files)} files")
    else:
        print(f"  ✗ Sparse reconstruction: Not found")
    
    # Check dense reconstruction
    if os.path.exists(dense_path):
        size = os.path.getsize(dense_path)
        print(f"  ✓ Dense PLY: {size/1024/1024:.2f} MB")
    else:
        print(f"  ○ Dense PLY: Not generated (sparse only)")

def main():
    print("images_to_play Preset Test Suite")
    print("="*60)
    
    # Get test images
    files = get_image_files()
    print(f"Found {len(files)} test images")
    
    if len(files) == 0:
        print("No test images found!")
        sys.exit(1)
    
    # Test presets
    presets = ['low', 'medium', 'high']
    results = {}
    
    for preset in presets:
        job_id = upload_and_process(preset, files, max_files=20)
        if job_id:
            status = monitor_job(job_id, preset, timeout=300)
            results[preset] = {
                'job_id': job_id,
                'status': status
            }
            check_output(job_id, preset)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for preset, result in results.items():
        status = result.get('status', {})
        print(f"{preset.upper()}: {status.get('status', 'unknown')} | Job ID: {result['job_id']}")

if __name__ == '__main__':
    main()
