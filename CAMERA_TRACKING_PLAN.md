# Camera Tracking Integration Plan

> Based on: Polyfjord's automated photogrammetry tracking workflow
> Source: https://youtube.com/polyfjord

---

## What the Polyfjord Script Does (Step by Step)

The script automates 4 steps per video:

```
Video → FFmpeg frames → COLMAP feature_extractor → COLMAP sequential_matcher → COLMAP mapper → TXT export
```

### Step 1: Frame Extraction (FFmpeg)
```bat
ffmpeg -loglevel error -stats -i "video.mp4" -qscale:v 2 "frame_%06d.jpg"
```
- Extracts **every frame** (no interval — dense sampling for tracking)
- `-qscale:v 2` = high quality JPEG (lower = better, 2 is near-lossless)
- No frame skipping — COLMAP's sequential matcher handles redundancy

**Current codebase:** `app.py:368-432` uses OpenCV (`cv2.VideoCapture`) with frame_interval.
**Difference:** OpenCV is 2-5x slower than FFmpeg for extraction. Polyfjord uses FFmpeg.

### Step 2: Feature Extraction (COLMAP)
```bat
colmap feature_extractor ^
    --database_path "database.db" ^
    --image_path "images/" ^
    --ImageReader.single_camera 1 ^
    --SiftExtraction.use_gpu 1 ^
    --SiftExtraction.max_image_size 4096
```
- `single_camera 1` — assumes all frames from same camera (true for video)
- `max_image_size 4096` — limits resolution for feature extraction
- Uses GPU for SIFT features

**Current codebase:** `run_glomap.py` already does this. Same flags.
**Difference:** None — this step is already compatible.

### Step 3: Sequential Matching (COLMAP) — KEY DIFFERENCE
```bat
colmap sequential_matcher ^
    --database_path "database.db" ^
    --SequentialMatching.overlap 15
```
- Uses **sequential_matcher** instead of exhaustive_matcher
- `overlap 15` — each frame matches against the next 15 frames
- **Much faster** than exhaustive for video (O(n) vs O(n²))
- **Much better** for video — exploits temporal ordering

**Current codebase:** `run_glomap.py` uses `exhaustive_matcher` or `sequential_matcher` with overlap 30.
**Difference:** The current sequential matcher uses overlap 30 (from the preset). Polyfjord uses 15. For camera tracking, 15 is sufficient — we don't need as many matches as 3D reconstruction.

### Step 4: Sparse Reconstruction (COLMAP mapper)
```bat
colmap mapper ^
    --database_path "database.db" ^
    --image_path "images/" ^
    --output_path "sparse/" ^
    --Mapper.num_threads %NUMBER_OF_PROCESSORS%
```
- Uses all CPU threads
- No special flags — default mapper settings

**Current codebase:** `run_glomap.py` does this with many more flags (tri_angle, reproj_error, etc.).
**Difference:** For camera tracking, we want the default (lenient) settings — keep more points for better camera pose estimation. The current presets are tuned for 3D reconstruction quality, not tracking.

### Step 5: TXT Export (COLMAP model_converter)
```bat
colmap model_converter ^
    --input_path "sparse/0" ^
    --output_path "sparse/" ^
    --output_type TXT
```
- Exports the reconstruction as **text files**
- Creates 3 files:
  - `cameras.txt` — camera intrinsics (focal length, distortion)
  - `images.txt` — camera poses (position + rotation per frame)
  - `points3D.txt` — sparse 3D points

**Current codebase:** NOT implemented. The app only exports PLY files.
**This is the critical missing piece** — the TXT files are what Blender reads.

---

## What We Need to Add

### 1. FFmpeg Frame Extraction (replace OpenCV for video)

The current `extract_frames_from_video()` uses OpenCV. FFmpeg is faster and more reliable.

```python
def extract_frames_ffmpeg(video_path, output_folder, ffmpeg_path='ffmpeg'):
    """Extract all frames from video using FFmpeg (faster than OpenCV)."""
    os.makedirs(output_folder, exist_ok=True)
    
    cmd = [
        ffmpeg_path,
        '-loglevel', 'error',
        '-stats',
        '-i', video_path,
        '-qscale:v', '2',
        os.path.join(output_folder, 'frame_%06d.jpg')
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    # Count extracted frames
    frames = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    return len(frames)
```

**Why FFmpeg over OpenCV:**
- 2-5x faster frame extraction
- Better codec support (handles all video formats)
- No need to decode frame-by-frame in Python
- Polyfjord uses it — proven workflow

**FFmpeg location on this system:**
```
C:\Users\f\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe
```

### 2. Camera Tracking Preset

A new preset optimized for tracking (not reconstruction):

```json
{
    "name": "camera_tracking",
    "features": 8192,
    "peak": 0.008,
    "tracks": 500000,
    "octaves": 4,
    "tri_angle": 0.5,
    "reproj_error": 8.0,
    "match_ratio": 0.85,
    "matcher": "sequential_matcher",
    "sequential_overlap": 15,
    "dense": false,
    "export_txt": true,
    "description": "Camera tracking — poses only, no dense reconstruction, exports TXT for Blender"
}
```

**Key differences from 3D reconstruction presets:**
- `sequential_matcher` with `overlap 15` (not exhaustive)
- `dense: false` — skip MVS entirely (we only need camera poses)
- `export_txt: true` — export cameras.txt, images.txt, points3D.txt
- Lower feature count (8192 vs 32768) — faster, sufficient for tracking
- Fewer octaves (4 vs 5) — video frames are similar, don't need scale space

### 3. TXT Export Command

After sparse reconstruction, run:

```python
def export_colmap_txt(colmap_path, sparse_path, output_path):
    """Export COLMAP reconstruction as TXT (cameras.txt, images.txt, points3D.txt)."""
    cmd = [
        colmap_path, 'model_converter',
        '--input_path', sparse_path,
        '--output_path', output_path,
        '--output_type', 'TXT'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"model_converter failed: {result.stderr}")
```

### 4. Camera Pose Extraction from TXT

Parse `images.txt` to get camera poses:

```python
def parse_colmap_images_txt(txt_path):
    """Parse COLMAP images.txt into camera pose data.
    
    Returns list of dicts sorted by frame number:
    [
        {
            'name': 'frame_000001.jpg',
            'frame': 0,
            'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,  # rotation quaternion
            'tx': 0.0, 'ty': 0.0, 'tz': 0.0,  # translation vector
            'camera_id': 1,
        },
        ...
    ]
    
    images.txt format (two lines per image):
    Line 1: IMAGE_ID QW QX QQY QZ TX TY TZ CAMERA_ID NAME
    Line 2: POINTS2D[] as (X, Y, POINT3D_ID) pairs
    """
    poses = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) < 10:
                continue
            
            # This is line 1 of an image entry
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]
            
            # Extract frame number from filename
            frame_num = 0
            import re
            match = re.search(r'(\d+)', name)
            if match:
                frame_num = int(match.group(1))
            
            poses.append({
                'name': name,
                'frame': frame_num,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'camera_id': camera_id,
            })
    
    poses.sort(key=lambda p: p['frame'])
    return poses
```

### 5. Camera Intrinsics from TXT

Parse `cameras.txt` to get focal length, distortion:

```python
def parse_colmap_cameras_txt(txt_path):
    """Parse COLMAP cameras.txt into camera intrinsic data.
    
    cameras.txt format:
    CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    
    Models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV
    """
    cameras = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params,
                'fx': params[0] if model in ('PINHOLE', 'OPENCV') else params[0],
                'fy': params[1] if model in ('PINHOLE', 'OPENCV') else params[0],
                'cx': params[2] if model in ('PINHOLE', 'OPENCV') else params[1],
                'cy': params[3] if model in ('PINHOLE', 'OPENCV') else params[2],
            }
    
    return cameras
```

### 6. FBX Export (Camera Animation)

Two approaches:

#### Option A: GLTF Export (recommended, no extra dependencies)
```python
import json
import numpy as np
from scipy.spatial.transform import Rotation

def export_camera_gltf(poses, cameras, output_path, fps=30):
    """Export camera animation as GLTF 2.0 file.
    
    GLTF is widely supported (Blender, Unity, Unreal, web viewers).
    No external dependencies needed — just write JSON.
    """
    # Build keyframe data
    timestamps = []
    positions = []
    rotations = []
    
    for i, pose in enumerate(poses):
        timestamps.append(i / fps)
        
        # COLMAP gives camera-from-world transform
        # We need world-from-camera (camera position in world space)
        q = [pose['qx'], pose['qy'], pose['qz'], pose['qw']]  # scipy format: xyzw
        R = Rotation.from_quat(q).as_matrix()
        t = np.array([pose['tx'], pose['ty'], pose['tz']])
        
        # Invert: camera position = -R^T @ t
        cam_pos = -R.T @ t
        cam_rot = Rotation.from_matrix(R.T)  # Transpose = inverse for rotation matrices
        
        positions.append(cam_pos.tolist())
        rotations.append(cam_rot.as_quat().tolist())  # xyzw
    
    # Build GLTF structure
    gltf = {
        "asset": {"version": "2.0", "generator": "images_to_play"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{
            "name": "Camera",
            "camera": 0,
            "translation": positions[0],
            "rotation": rotations[0],
        }],
        "cameras": [{
            "type": "perspective",
            "perspective": {
                "aspectRatio": cameras[1]['width'] / cameras[1]['height'],
                "yfov": 2 * np.arctan(cameras[1]['height'] / (2 * cameras[1]['fy'])),
                "znear": 0.01,
            }
        }],
        "animations": [{
            "channels": [{
                "sampler": 0,
                "target": {"node": 0, "path": "translation"}
            }, {
                "sampler": 1,
                "target": {"node": 0, "path": "rotation"}
            }],
            "samplers": [
                {"input": 0, "output": 1},  # translation
                {"input": 0, "output": 2},  # rotation
            ]
        }],
        "accessors": [...],  # timestamp, position, rotation buffer views
        "bufferViews": [...],
        "buffers": [...],
    }
    
    # Write as .gltf (JSON) or .glb (binary)
    with open(output_path, 'w') as f:
        json.dump(gltf, f, indent=2)
```

#### Option B: FBX Export (requires Autodesk FBX SDK)
```python
# Only if FBX SDK is installed
# pip install fbx  (or install Autodesk FBX SDK)
try:
    import fbx
    FBX_AVAILABLE = True
except ImportError:
    FBX_AVAILABLE = False

def export_camera_fbx(poses, cameras, output_path, fps=30):
    """Export camera animation as FBX file."""
    if not FBX_AVAILABLE:
        raise ImportError("FBX SDK not installed. Use GLTF export instead.")
    
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "CameraTracking")
    
    # Create camera
    camera = fbx.FbxCamera.Create(scene, "TrackingCamera")
    camera.SetAspect(fbx.FbxCamera.eFixedResolution)
    camera.SetResolutionMode(fbx.FbxCamera.eFixedResolution)
    camera.FrameWidth.Set(cameras[1]['width'])
    camera.FrameHeight.Set(cameras[1]['height'])
    
    # Set focal length
    sensor_width = 36.0  # mm (standard full frame)
    focal_length_mm = cameras[1]['fx'] * sensor_width / cameras[1]['width']
    camera.SetApertureWidth(sensor_width)
    camera.FocalLength.Set(focal_length_mm)
    
    # Create animation stack and layer
    anim_stack = fbx.FbxAnimStack.Create(scene, "CameraAnimation")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    anim_stack.AddMember(anim_layer)
    
    # Add keyframes
    time_mode = fbx.FbxTime.eFrames30
    fbx.FbxTime.SetGlobalTimeMode(time_mode)
    
    for i, pose in enumerate(poses):
        time = fbx.FbxTime()
        time.SetFrame(i, time_mode)
        
        # Convert COLMAP pose to world coordinates
        q = [pose['qx'], pose['qy'], pose['qz'], pose['qw']]
        R = Rotation.from_quat(q).as_matrix()
        t = np.array([pose['tx'], pose['ty'], pose['tz']])
        cam_pos = -R.T @ t
        
        # Set position
        camera.LclTranslation.Set(fbx.FbxDouble3(cam_pos[0], cam_pos[1], cam_pos[2]))
        camera.LclTranslation.GetCurveNode(anim_layer, True)
        
        # Set rotation (convert to Euler)
        from scipy.spatial.transform import Rotation as R
        euler = R.from_matrix(R.T).as_euler('XYZ', degrees=True)
        camera.LclRotation.Set(fbx.FbxDouble3(euler[0], euler[1], euler[2]))
        camera.LclRotation.GetCurveNode(anim_layer, True)
    
    # Export
    exporter = fbx.FbxExporter.Create(manager, "")
    exporter.Initialize(output_path, -1, manager.GetIOSettings())
    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
```

### 7. FBX Export (Primary Target)

FBX is the industry standard for camera animation data. Use Autodesk FBX SDK.

```python
def export_camera_fbx(poses, cameras, output_path, fps=30):
    """Export camera animation as FBX file.
    
    Creates an animated camera with position/rotation keyframes.
    Compatible with: Blender, Maya, 3ds Max, Unreal Engine, Unity, Houdini.
    """
    import fbx
    import numpy as np
    from scipy.spatial.transform import Rotation
    
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "CameraTracking")
    
    # Create camera
    camera = fbx.FbxCamera.Create(scene, "TrackingCamera")
    camera.SetAspect(fbx.FbxCamera.eFixedResolution)
    camera.FrameWidth.Set(cameras[1]['width'])
    camera.FrameHeight.Set(cameras[1]['height'])
    
    # Focal length in mm (sensor width = 36mm standard full frame)
    sensor_width = 36.0
    focal_length_mm = cameras[1]['fx'] * sensor_width / cameras[1]['width']
    camera.SetApertureWidth(sensor_width)
    camera.FocalLength.Set(focal_length_mm)
    
    # Animation stack
    anim_stack = fbx.FbxAnimStack.Create(scene, "CameraAnimation")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    anim_stack.AddMember(anim_layer)
    
    # Set frame rate
    time_mode = fbx.FbxTime.eFrames30
    fbx.FbxTime.SetGlobalTimeMode(time_mode)
    
    # Add keyframes
    for i, pose in enumerate(poses):
        time = fbx.FbxTime()
        time.SetFrame(i, time_mode)
        
        # COLMAP gives camera-from-world; invert to get world-from-camera
        q = [pose['qx'], pose['qy'], pose['qz'], pose['qw']]
        R = Rotation.from_quat(q).as_matrix()
        t = np.array([pose['tx'], pose['ty'], pose['tz']])
        cam_pos = -R.T @ t
        cam_euler = Rotation.from_matrix(R.T).as_euler('XYZ', degrees=True)
        
        # Keyframe position
        pos_curve_x = camera.LclTranslation.GetCurve(anim_layer, "X", True)
        pos_curve_y = camera.LclTranslation.GetCurve(anim_layer, "Y", True)
        pos_curve_z = camera.LclTranslation.GetCurve(anim_layer, "Z", True)
        pos_curve_x.KeySet(pos_curve_x.KeyAdd(time), time, cam_pos[0])
        pos_curve_y.KeySet(pos_curve_y.KeyAdd(time), time, cam_pos[1])
        pos_curve_z.KeySet(pos_curve_z.KeyAdd(time), time, cam_pos[2])
        
        # Keyframe rotation
        rot_curve_x = camera.LclRotation.GetCurve(anim_layer, "X", True)
        rot_curve_y = camera.LclRotation.GetCurve(anim_layer, "Y", True)
        rot_curve_z = camera.LclRotation.GetCurve(anim_layer, "Z", True)
        rot_curve_x.KeySet(rot_curve_x.KeyAdd(time), time, cam_euler[0])
        rot_curve_y.KeySet(rot_curve_y.KeyAdd(time), time, cam_euler[1])
        rot_curve_z.KeySet(rot_curve_z.KeyAdd(time), time, cam_euler[2])
    
    # Export
    exporter = fbx.FbxExporter.Create(manager, "")
    exporter.Initialize(output_path, -1, manager.GetIOSettings())
    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
```

### 8. GLTF Export (Fallback, No Extra Dependencies)

If FBX SDK isn't installed, export as GLTF — widely supported, pure Python.

```python
def export_camera_gltf(poses, cameras, output_path, fps=30):
    """Export camera animation as GLTF 2.0 (JSON). No external dependencies."""
    import struct
    import numpy as np
    from scipy.spatial.transform import Rotation
    
    timestamps = []
    positions = []
    rotations = []
    
    for i, pose in enumerate(poses):
        timestamps.append(i / fps)
        
        q = [pose['qx'], pose['qy'], pose['qz'], pose['qw']]
        R = Rotation.from_quat(q).as_matrix()
        t = np.array([pose['tx'], pose['ty'], pose['tz']])
        cam_pos = -R.T @ t
        cam_rot = Rotation.from_matrix(R.T).as_quat()  # xyzw
        
        positions.append(cam_pos.tolist())
        rotations.append(cam_rot.tolist())
    
    # Pack binary data
    ts_bytes = struct.pack(f'{len(timestamps)}f', *timestamps)
    pos_bytes = struct.pack(f'{len(positions)*3}f', *[c for p in positions for c in p])
    rot_bytes = struct.pack(f'{len(rotations)*4}f', *[c for r in rotations for c in r])
    bin_data = ts_bytes + pos_bytes + rot_bytes
    
    # Offsets
    ts_offset = 0
    pos_offset = len(ts_bytes)
    rot_offset = pos_offset + len(pos_bytes)
    
    gltf = {
        "asset": {"version": "2.0", "generator": "images_to_play"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Camera", "camera": 0}],
        "cameras": [{
            "type": "perspective",
            "perspective": {
                "aspectRatio": cameras[1]['width'] / cameras[1]['height'],
                "yfov": 2 * np.arctan(cameras[1]['height'] / (2 * cameras[1]['fy'])),
                "znear": 0.01
            }
        }],
        "animations": [{
            "channels": [
                {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                {"sampler": 1, "target": {"node": 0, "path": "rotation"}}
            ],
            "samplers": [
                {"input": 0, "output": 1},
                {"input": 0, "output": 2}
            ]
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(timestamps), "type": "SCALAR", "min": [min(timestamps)], "max": [max(timestamps)]},
            {"bufferView": 1, "componentType": 5126, "count": len(positions), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(rotations), "type": "VEC4"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": ts_offset, "byteLength": len(ts_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": rot_offset, "byteLength": len(rot_bytes), "target": 34962}
        ],
        "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(bin_data).decode(), "byteLength": len(bin_data)}]
    }
    
    import base64
    gltf["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(bin_data).decode()
    
    with open(output_path, 'w') as f:
        json.dump(gltf, f, indent=2)
```

---

## Integration into the Existing Codebase

### New File: `camera_tracking.py`

A new module that orchestrates the camera tracking pipeline:

```
Video Input
    ↓
extract_frames_ffmpeg()          ← NEW (replaces OpenCV for video)
    ↓
run_colmap() with tracking preset ← MODIFIED (new preset, sequential matcher)
    ↓
export_colmap_txt()              ← NEW (model_converter → TXT)
    ↓
parse_colmap_images_txt()        ← NEW (extract camera poses)
    ↓
export_camera_fbx()              ← NEW (FBX — primary, industry standard)
export_camera_gltf()             ← NEW (GLTF — fallback, no dependencies)
    ↓
Package results as ZIP           ← NEW (all formats in one download)
```

### Changes to `app.py`

1. Add "Camera Tracking" as a third method option (alongside Multi-Image and ML-Sharp)
2. New `/track` endpoint that accepts video upload
3. New `/track/<job_id>/download` endpoint for downloading the tracking package

### Changes to `run_glomap.py`

1. Add `camera_tracking` preset to the preset system
2. Add `export_colmap_txt()` function
3. When tracking preset is selected, skip dense reconstruction and MVS

### Changes to `index.html`

1. Add "Camera Tracking" radio button in the method selector (alongside Multi-Image and ML-Sharp)
2. Show tracking-specific settings (frame rate, overlap)
3. Show tracking results: download links for FBX, GLTF, and COLMAP TXT files
4. Preview: show extracted frames timeline and camera path visualization

### Output Package Structure

```
tracking_output/
├── camera_poses.json       ← All camera poses (for programmatic use)
├── cameras.txt             ← COLMAP format (for Blender Photogrammetry Importer)
├── images.txt              ← COLMAP format
├── points3D.txt            ← COLMAP format
├── sparse.ply              ← Sparse point cloud (for reference)
├── camera_animation.fbx    ← FBX camera animation (primary — Maya, 3ds Max, Unreal, Unity, Blender)
└── camera_animation.glTF   ← GLTF camera animation (fallback — Blender, Unity, web viewers)
```

### Usage in 3D Software

| Software | Import Method |
|----------|--------------|
| **Blender** | File → Import → FBX (.fbx) or glTF 2.0 (.gltf) |
| **Maya** | File → Import → FBX |
| **3ds Max** | File → Import → FBX |
| **Unreal Engine** | Drag & drop FBX into Content Browser |
| **Unity** | Drag & drop FBX into Assets |
| **Houdini** | File → Import → FBX |

---

## Key Technical Details

### COLMAP Coordinate System
- COLMAP uses **camera-from-world** transforms (inverse of what most 3D software expects)
- To get the camera position in world space: `cam_pos = -R^T @ t`
- To get the camera rotation in world space: `cam_rot = R^T`
- This inversion is critical — get it wrong and the camera path is mirrored/offset

### COLMAP Quaternion Convention
- COLMAP stores quaternions as `(qw, qx, qy, qz)` — scalar first
- Blender uses `(qw, qx, qy, qz)` — same convention
- scipy uses `(qx, qy, qz, qw)` — vector first (different!)
- Always check the convention when converting

### Frame Rate Handling
- COLMAP doesn't know about frame rates — it just processes images
- The frame rate is only needed for the animation timeline
- Store the source video FPS in the output metadata
- Default to 30fps if unknown

### Sequential Matcher Overlap
- `overlap 15` means each frame matches against the next 15 frames
- For 30fps video, this is 0.5 seconds of look-ahead
- Higher overlap = more robust tracking but slower
- For fast camera motion, increase to 20-30
- For slow/steady video, 10-15 is sufficient

### Loop Closure for Long Videos
For videos > 2 minutes, consider adding vocab_tree matching:
```bat
colmap vocab_tree_matcher ^
    --database_path "database.db" ^
    --VocabTreeMatching.vocab_tree_path "vocab_tree_flickr100K.bin"
```
This helps when the camera returns to a previously seen location (loop closure).

---

## What Already Works in the Codebase

| Component | Status | Notes |
|-----------|--------|-------|
| COLMAP feature_extractor | **DONE** | `run_glomap.py` — works as-is |
| COLMAP sequential_matcher | **DONE** | `run_glomap.py` — works, but overlap 30 (not 15) |
| COLMAP mapper | **DONE** | `run_glomap.py` — works with extra flags |
| GPU assignment | **DONE** | `app.py` — multi-GPU support exists |
| Progress streaming | **DONE** | `app.py` — SSE for real-time updates |
| Job management | **DONE** | `app.py` — create, monitor, cancel, cleanup |
| Video upload | **DONE** | `app.py:1568-1588` — handles video files |
| Frame extraction | **PARTIAL** | Uses OpenCV, should use FFmpeg |
| COLMAP model_converter | **NOT DONE** | Need to add TXT export |
| Camera pose parsing | **NOT DONE** | Need to parse images.txt |
| FBX export | **NOT DONE** | New module needed |
| GLTF export | **NOT DONE** | New module needed (fallback) |
| UI for tracking mode | **NOT DONE** | New frontend section needed |

---

## Estimated Implementation Time

| Step | Effort | Description |
|------|--------|-------------|
| FFmpeg extraction | 30 min | Replace OpenCV with FFmpeg subprocess |
| Tracking preset JSON | 10 min | New preset file with sequential_matcher, overlap 15 |
| TXT export function | 15 min | model_converter subprocess call |
| Camera pose parser | 30 min | Parse images.txt, cameras.txt |
| FBX export | 2 hr | FBX SDK camera animation with keyframes |
| GLTF export | 1.5 hr | GLTF fallback (no dependencies) |
| App.py integration | 1 hr | New endpoint, method selector |
| Frontend UI | 1.5 hr | Camera tracking mode, download UI |
| Testing | 1 hr | Test with real video |
| **Total** | **~8 hr** | Full implementation |
