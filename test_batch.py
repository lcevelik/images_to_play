"""
Batch Processing API Test for images_to_play
Demonstrates how to use the batch processing endpoints.
"""
import requests
import time
import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:5000"
SOURCE_DIR = r"F:\Codebase\images_to_play\simple_splat\App\processing\8e12c030-af50-447e-a31c-f3d0cc856b2c\source"

def get_images(max_count=10):
    """Get a subset of test images."""
    files = []
    for f in sorted(os.listdir(SOURCE_DIR)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            files.append(os.path.join(SOURCE_DIR, f))
            if len(files) >= max_count:
                break
    return files

def test_batch_api():
    """Test the batch processing API."""
    print("Batch Processing API Test")
    print("=" * 60)
    
    # 1. Create batch
    print("\n1. Creating batch...")
    resp = requests.post(f"{BASE_URL}/batch/create", json={'name': 'Test Batch'})
    batch_data = resp.json()
    batch_id = batch_data['batch_id']
    print(f"   Created batch: {batch_id}")
    
    # 2. Add jobs with different presets
    images = get_images(10)
    print(f"\n2. Adding jobs with {len(images)} images each...")
    
    for preset in ['low', 'medium', 'high']:
        upload_files = []
        for fpath in images:
            fname = os.path.basename(fpath)
            upload_files.append(('files', (fname, open(fpath, 'rb'), 'image/jpeg')))
        
        data = {
            'name': f'Test {preset}',
            'preset': preset,
            'matcher_type': 'exhaustive_matcher',
            'interval': '1',
            'quality_scale': 'standard',
            'trainer': 'brush',
        }
        
        try:
            resp = requests.post(
                f"{BASE_URL}/batch/{batch_id}/add",
                files=upload_files,
                data=data
            )
            result = resp.json()
            print(f"   Added {preset}: {result.get('job_id', 'error')}")
        finally:
            for _, (_, fh, _) in upload_files:
                fh.close()
    
    # 3. Check batch status before starting
    print("\n3. Batch status before start:")
    resp = requests.get(f"{BASE_URL}/batch/{batch_id}/status")
    status = resp.json()
    print(f"   Total jobs: {status['total']}")
    print(f"   Status: {status['status']}")
    
    # 4. Start batch
    print("\n4. Starting batch...")
    resp = requests.post(f"{BASE_URL}/batch/{batch_id}/start")
    print(f"   Started: {resp.json()}")
    
    # 5. Monitor progress
    print("\n5. Monitoring progress (checking every 10s)...")
    for i in range(60):  # Monitor for up to 10 minutes
        time.sleep(10)
        resp = requests.get(f"{BASE_URL}/batch/{batch_id}/status")
        status = resp.json()
        
        print(f"   [{i*10}s] Status: {status['status']} | "
              f"Complete: {status['completed']}/{status['total']} | "
              f"Running: {status['running']} | "
              f"Progress: {status['overall_progress']:.1f}%")
        
        if status['status'] in ('complete', 'failed', 'partial'):
            break
    
    # 6. Final status
    print("\n6. Final status:")
    resp = requests.get(f"{BASE_URL}/batch/{batch_id}/status")
    final = resp.json()
    print(json.dumps(final, indent=2, default=str))
    
    # 7. List all batches
    print("\n7. All batches:")
    resp = requests.get(f"{BASE_URL}/batch/list")
    print(json.dumps(resp.json(), indent=2, default=str))

def test_folder_upload():
    """Test folder-based batch upload."""
    print("\n" + "=" * 60)
    print("Testing folder-based batch upload...")
    print("=" * 60)
    
    # Create a test ZIP with subfolders
    import zipfile
    import tempfile
    
    zip_path = os.path.join(tempfile.gettempdir(), 'test_batch.zip')
    images = get_images(5)
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Create 2 "scenes" with 5 images each
        for scene_num in range(1, 3):
            for i, img_path in enumerate(images):
                fname = f'scene_{scene_num}/img_{i:03d}.jpg'
                zf.write(img_path, fname)
    
    print(f"Created test ZIP: {zip_path}")
    
    # Upload ZIP
    with open(zip_path, 'rb') as f:
        resp = requests.post(
            f"{BASE_URL}/batch/upload-folder",
            files={'file': ('test_batch.zip', f, 'application/zip')},
            data={'name': 'Folder Test', 'preset': 'low'}
        )
    
    result = resp.json()
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if 'batch_id' in result:
        batch_id = result['batch_id']
        print(f"\nStarting batch {batch_id}...")
        resp = requests.post(f"{BASE_URL}/batch/{batch_id}/start")
        print(f"Started: {resp.json()}")
    
    # Cleanup
    os.remove(zip_path)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'folder':
        test_folder_upload()
    else:
        test_batch_api()
