"""
Camera Tracking & Matchmoving Module for images_to_play

Extracts camera poses from COLMAP reconstruction and exports as:
- FBX (via pyfbx or custom writer)
- GLTF (via trimesh)
- JSON (universal format)
- Blender Python import script

Usage:
    from camera_tracking import extract_camera_poses, export_camera_fbx, export_camera_gltf
    
    poses = extract_camera_poses("path/to/sparse/0")
    export_camera_gltf(poses, "camera.gltf", fps=30)
    export_camera_fbx(poses, "camera.fbx", fps=30)
"""

import os
import json
import math
import struct
import numpy as np


def extract_camera_poses(sparse_dir, image_dir=None):
    """Extract camera poses from COLMAP reconstruction.
    
    Args:
        sparse_dir: Path to COLMAP sparse reconstruction (e.g., sparse/0/)
        image_dir: Optional path to images folder (for frame ordering)
    
    Returns:
        List of dicts sorted by frame order:
        [
            {
                'frame': 0,
                'name': 'frame_00001.jpg',
                'position': [x, y, z],        # camera position in world space
                'rotation_matrix': [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]],
                'rotation_quat': [w, x, y, z],  # quaternion (wxyz)
                'focal_length': fx,
                'width': 1920,
                'height': 1080,
                'timestamp': 0.033,            # seconds (based on frame rate)
            },
            ...
        ]
    """
    import pycolmap
    
    recon = pycolmap.Reconstruction(sparse_dir)
    
    poses = []
    for image_id, image in recon.images.items():
        cam = recon.cameras[image.camera_id]
        
        # pycolmap gives cam_from_world (world-to-camera) as 3x4
        w2c_3x4 = np.array(image.cam_from_world().matrix(), dtype=np.float64)
        
        # Invert to get world-from-camera (camera-to-world)
        R = w2c_3x4[:3, :3]
        t = w2c_3x4[:3, 3]
        
        # c2w = inverse of w2c
        R_c2w = R.T
        t_c2w = -R.T @ t
        
        # Convert rotation matrix to quaternion (wxyz format)
        quat = _rotation_matrix_to_quaternion(R_c2w)
        
        poses.append({
            'name': image.name,
            'position': t_c2w.tolist(),
            'rotation_matrix': R_c2w.tolist(),
            'rotation_quat': [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            'focal_length': float(cam.focal_length_x),
            'width': cam.width,
            'height': cam.height,
        })
    
    # Sort by filename (which should be frame order for video extraction)
    poses.sort(key=lambda p: p['name'])
    
    # Add frame numbers and timestamps (assume 30fps default)
    for i, pose in enumerate(poses):
        pose['frame'] = i
        pose['timestamp'] = i / 30.0  # default 30fps
    
    return poses


def _rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def _quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def export_camera_json(poses, output_path, fps=30):
    """Export camera poses as JSON (universal format).
    
    Args:
        poses: List from extract_camera_poses()
        output_path: Output .json file path
        fps: Frame rate for timestamp calculation
    """
    # Update timestamps with correct fps
    for i, pose in enumerate(poses):
        pose['timestamp'] = i / fps
    
    data = {
        'version': '1.0',
        'source': 'images_to_play',
        'fps': fps,
        'frame_count': len(poses),
        'duration': len(poses) / fps,
        'cameras': poses
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path


def export_camera_gltf(poses, output_path, fps=30):
    """Export camera animation as GLTF/GLB file.
    
    Args:
        poses: List from extract_camera_poses()
        output_path: Output .gltf file path
        fps: Frame rate
    """
    try:
        import trimesh
        import trimesh.scene
    except ImportError:
        # Fallback: write raw GLTF JSON
        return _export_gltf_manual(poses, output_path, fps)
    
    # Build a scene with animated camera
    scene = trimesh.Scene()
    
    # Add a camera node with animation
    # GLTF uses column-major matrices
    # We'll build the GLTF manually for precise control
    return _export_gltf_manual(poses, output_path, fps)


def _export_gltf_manual(poses, output_path, fps=30):
    """Write GLTF file manually with camera animation."""
    import base64
    
    num_frames = len(poses)
    
    # Build translation and rotation arrays
    translations = []
    rotations = []
    for pose in poses:
        pos = pose['position']
        quat = pose['rotation_quat']  # [w, x, y, z]
        # GLTF expects [x, y, z, w]
        translations.extend([pos[0], pos[1], pos[2]])
        rotations.extend([quat[1], quat[2], quat[3], quat[0]])
    
    # Pack as binary data
    trans_bytes = struct.pack(f'<{len(translations)}f', *translations)
    rot_bytes = struct.pack(f'<{len(rotations)}f', *rotations)
    
    # Combine into a single buffer
    trans_offset = 0
    trans_len = len(trans_bytes)
    rot_offset = trans_len
    rot_len = len(rot_bytes)
    total_buffer_len = trans_len + rot_len
    buffer_data = trans_bytes + rot_bytes
    buffer_b64 = base64.b64encode(buffer_data).decode('ascii')
    
    # Time accessor
    times = [i / fps for i in range(num_frames)]
    time_bytes = struct.pack(f'<{len(times)}f', *times)
    time_offset = 0
    time_len = len(time_bytes)
    
    # Build the full buffer with time data
    full_buffer = time_bytes + buffer_data
    buffer_b64 = base64.b64encode(full_buffer).decode('ascii')
    
    # Adjust offsets
    time_offset = 0
    trans_offset = time_len
    rot_offset = time_len + trans_len
    
    gltf = {
        "asset": {"version": "2.0", "generator": "images_to_play"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": "Camera",
                "camera": 0,
                "translation": poses[0]['position'],
                "rotation": poses[0]['rotation_quat'][1:] + [poses[0]['rotation_quat'][0]],  # xyzw
            }
        ],
        "cameras": [
            {
                "type": "perspective",
                "perspective": {
                    "aspectRatio": poses[0]['width'] / poses[0]['height'] if poses[0]['height'] > 0 else 16/9,
                    "yfov": 2 * math.atan(poses[0]['height'] / (2 * poses[0]['focal_length'])) if poses[0]['focal_length'] > 0 else 1.0,
                    "znear": 0.01,
                    "zfar": 1000.0
                }
            }
        ],
        "animations": [
            {
                "name": "CameraAnimation",
                "channels": [
                    {
                        "sampler": 0,
                        "target": {"node": 0, "path": "translation"}
                    },
                    {
                        "sampler": 1,
                        "target": {"node": 0, "path": "rotation"}
                    }
                ],
                "samplers": [
                    {
                        "input": 0,
                        "output": 1,
                        "interpolation": "LINEAR"
                    },
                    {
                        "input": 0,
                        "output": 2,
                        "interpolation": "LINEAR"
                    }
                ]
            }
        ],
        "accessors": [
            # 0: time
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": num_frames,
                "type": "SCALAR",
                "min": [times[0]],
                "max": [times[-1]]
            },
            # 1: translations
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": num_frames,
                "type": "VEC3",
                "min": [min(t[i] for t in [p['position'] for p in poses]) for i in range(3)],
                "max": [max(t[i] for t in [p['position'] for p in poses]) for i in range(3)]
            },
            # 2: rotations
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": num_frames,
                "type": "VEC4"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": time_offset, "byteLength": time_len, "target": 0},
            {"buffer": 0, "byteOffset": trans_offset, "byteLength": trans_len, "target": 0},
            {"buffer": 0, "byteOffset": rot_offset, "byteLength": rot_len, "target": 0}
        ],
        "buffers": [
            {
                "uri": f"data:application/octet-stream;base64,{buffer_b64}",
                "byteLength": total_buffer_len + time_len
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(gltf, f, indent=2)
    
    return output_path


def export_camera_fbx(poses, output_path, fps=30):
    """Export camera animation as FBX file.
    
    Uses a minimal custom FBX binary writer (no external dependencies).
    FBX 7.4+ binary format.
    
    Args:
        poses: List from extract_camera_poses()
        output_path: Output .fbx file path
        fps: Frame rate
    """
    # FBX is complex binary — for now, export as ASCII FBX (widely compatible)
    return _export_fbx_ascii(poses, output_path, fps)


def _export_fbx_ascii(poses, output_path, fps=30):
    """Write FBX ASCII format with camera animation."""
    num_frames = len(poses)
    time_step = 1.0 / fps
    
    lines = []
    lines.append("; FBX 7.4.0 project file")
    lines.append("; Generated by images_to_play")
    lines.append("")
    lines.append("FBXHeaderExtension:  {")
    lines.append("    FBXHeaderVersion: 1003")
    lines.append("    FBXVersion: 7400")
    lines.append("    Creator: \"images_to_play camera tracker\"")
    lines.append("}")
    lines.append("")
    lines.append("Definitions:  {")
    lines.append("    Version: 100")
    lines.append("    Count: 2")
    lines.append("    ObjectType: \"Model\" {")
    lines.append("        Count: 1")
    lines.append("    }")
    lines.append("    ObjectType: \"AnimationStack\" {")
    lines.append("        Count: 1")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    
    # Objects
    lines.append("Objects:  {")
    lines.append("    Model: 1, \"Model::Camera\", \"Camera\" {")
    lines.append("        Version: 232")
    lines.append("        Properties70:  {")
    lines.append("            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\", 0, 0, 0")
    lines.append("            P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\", 0, 0, 0")
    lines.append("        }")
    lines.append("    }")
    lines.append("    AnimationStack: 0, \"AnimStack::CameraAnim\", \"\" {")
    lines.append("        Version: 100")
    lines.append("        Properties70:  {")
    lines.append("            P: \"LocalStart\", \"Lcl Translation\", \"\", \"A\", 0")
    lines.append("            P: \"LocalStop\", \"Lcl Translation\", \"\", \"A\", " + str(int(num_frames * time_step * 46186158400)) + "")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    
    # Connections
    lines.append("Connections:  {")
    lines.append("    Connect: \"OO\", 1, 0")
    lines.append("}")
    lines.append("")
    
    # Animation data (Takes section)
    lines.append("Takes:  {")
    lines.append("    Current: \"CameraAnim\"")
    lines.append("    Take: \"CameraAnim\" {")
    lines.append("        Version: 100")
    lines.append("        Channel: \"Transform\" {")
    lines.append("            Channel: \"T\" {")
    lines.append("                Channel: \"X\" {")
    
    # Translation X keyframes
    for i, pose in enumerate(poses):
        time_val = int(i * time_step * 46186158400)
        lines.append(f"                    Key: {time_val}, {pose['position'][0]}")
    
    lines.append("                }")
    lines.append("                Channel: \"Y\" {")
    for i, pose in enumerate(poses):
        time_val = int(i * time_step * 46186158400)
        lines.append(f"                    Key: {time_val}, {pose['position'][1]}")
    lines.append("                }")
    lines.append("                Channel: \"Z\" {")
    for i, pose in enumerate(poses):
        time_val = int(i * time_step * 46186158400)
        lines.append(f"                    Key: {time_val}, {pose['position'][2]}")
    lines.append("                }")
    lines.append("            }")
    lines.append("            Channel: \"R\" {")
    lines.append("                Channel: \"X\" {")
    
    # Convert quaternions to Euler angles for FBX
    for i, pose in enumerate(poses):
        q = pose['rotation_quat']
        euler = _quaternion_to_euler(q)
        time_val = int(i * time_step * 46186158400)
        lines.append(f"                    Key: {time_val}, {math.degrees(euler[0])}")
    
    lines.append("                }")
    lines.append("                Channel: \"Y\" {")
    for i, pose in enumerate(poses):
        q = pose['rotation_quat']
        euler = _quaternion_to_euler(q)
        time_val = int(i * time_step * 46186158400)
        lines.append(f"                    Key: {time_val}, {math.degrees(euler[1])}")
    lines.append("                }")
    lines.append("                Channel: \"Z\" {")
    for i, pose in enumerate(poses):
        q = pose['rotation_quat']
        euler = _quaternion_to_euler(q)
        time_val = int(i * time_step * 46186158400)
        lines.append(f"                    Key: {time_val}, {math.degrees(euler[2])}")
    lines.append("                }")
    lines.append("            }")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_path


def _quaternion_to_euler(q):
    """Convert quaternion [w, x, y, z] to Euler angles (XYZ order) in radians."""
    w, x, y, z = q
    
    # Roll (X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (Y)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1, min(1, sinp))
    pitch = math.asin(sinp)
    
    # Yaw (Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (roll, pitch, yaw)


def export_blender_script(poses, output_dir, fps=30):
    """Generate a Blender Python import script + JSON data file.
    
    Args:
        poses: List from extract_camera_poses()
        output_dir: Directory to write script + data
        fps: Frame rate
    
    Returns:
        Path to the Blender script
    """
    # Save pose data as JSON
    data_path = os.path.join(output_dir, 'camera_poses.json')
    export_camera_json(poses, data_path, fps)
    
    # Write Blender import script
    script_path = os.path.join(output_dir, 'import_camera.py')
    script = f'''"""
Blender Camera Import Script — Generated by images_to_play
Usage: In Blender, go to Scripting > Open > import_camera.py > Run Script

Requires: camera_poses.json in the same directory as this script
"""
import bpy
import json
import os
import mathutils

# Load pose data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'camera_poses.json')

with open(data_path) as f:
    data = json.load(f)

fps = data['fps']
poses = data['cameras']
num_frames = len(poses)

# Set frame rate
bpy.context.scene.render.fps = fps
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = num_frames

# Create camera
cam_data = bpy.data.cameras.new('TrackingCamera')
cam_obj = bpy.data.objects.new('TrackingCamera', cam_data)
bpy.context.scene.collection.objects.link(cam_obj)

# Set focal length from first frame
if poses[0]['focal_length'] > 0:
    # Convert pixel focal length to Blender mm (assuming 36mm sensor width)
    sensor_width = 36.0
    pixel_focal = poses[0]['focal_length']
    image_width = poses[0]['width'] if poses[0]['width'] > 0 else 1920
    cam_data.lens = (sensor_width * pixel_focal) / image_width

# Insert keyframes
for i, pose in enumerate(poses):
    frame = i + 1  # Blender frames are 1-indexed
    
    pos = pose['position']
    quat = pose['rotation_quat']  # [w, x, y, z]
    
    # Set transform
    cam_obj.location = (pos[0], pos[1], pos[2])
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = (quat[0], quat[1], quat[2], quat[3])
    
    # Insert keyframes
    cam_obj.keyframe_insert(data_path='location', frame=frame)
    cam_obj.keyframe_insert(data_path='rotation_quaternion', frame=frame)

# Set interpolation to linear for all keyframes
if cam_obj.animation_data and cam_obj.animation_data.action:
    for fcurve in cam_obj.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'LINEAR'

print(f"Imported {{num_frames}} camera keyframes at {{fps}} FPS")
print(f"Camera: TrackingCamera")
print(f"Frames: 1 - {{num_frames}}")
'''
    
    with open(script_path, 'w') as f:
        f.write(script)
    
    return script_path


def export_point_cloud_ply(sparse_dir, output_path, center=True):
    """Export sparse point cloud as PLY for reference in 3D software."""
    import pycolmap
    
    recon = pycolmap.Reconstruction(sparse_dir)
    
    points = []
    colors = []
    for pid, point in recon.points3D.items():
        points.append(point.xyz)
        colors.append(point.color)
    
    if not points:
        return None
    
    pts = np.array(points, dtype=np.float32)
    cols = np.array(colors, dtype=np.uint8)
    
    if center:
        centroid = pts.mean(axis=0)
        pts -= centroid
    
    # Write binary PLY
    with open(output_path, 'wb') as f:
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))
        
        vertex_dtype = np.dtype([('xyz', np.float32, 3), ('rgb', np.uint8, 3)])
        vertices = np.empty(len(pts), dtype=vertex_dtype)
        vertices['xyz'] = pts
        vertices['rgb'] = cols
        f.write(vertices.tobytes())
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Tracking — Extract and export camera poses from COLMAP")
    parser.add_argument('--input', '-i', required=True, help='COLMAP sparse reconstruction dir (sparse/0/)')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--format', '-f', choices=['json', 'gltf', 'fbx', 'blender', 'all'], default='all',
                        help='Export format (default: all)')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate (default: 30)')
    parser.add_argument('--pointcloud', action='store_true', help='Also export sparse point cloud as PLY')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Extracting camera poses from: {args.input}")
    poses = extract_camera_poses(args.input)
    print(f"Found {len(poses)} camera poses")
    
    if args.format in ('json', 'all'):
        path = export_camera_json(poses, os.path.join(args.output, 'cameras.json'), args.fps)
        print(f"  JSON: {path}")
    
    if args.format in ('gltf', 'all'):
        path = export_camera_gltf(poses, os.path.join(args.output, 'cameras.gltf'), args.fps)
        print(f"  GLTF: {path}")
    
    if args.format in ('fbx', 'all'):
        path = export_camera_fbx(poses, os.path.join(args.output, 'cameras.fbx'), args.fps)
        print(f"  FBX: {path}")
    
    if args.format in ('blender', 'all'):
        path = export_blender_script(poses, args.output, args.fps)
        print(f"  Blender script: {path}")
    
    if args.pointcloud:
        path = export_point_cloud_ply(args.input, os.path.join(args.output, 'sparse_cloud.ply'))
        if path:
            print(f"  Point cloud: {path}")
    
    print("Done!")
