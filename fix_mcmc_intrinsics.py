import os

p = r'F:\Codebase\images_to_play\simple_splat\App\gsplat_mcmc_trainer.py'
with open(p, 'r') as f:
    lines = f.readlines()

changes = 0

for i, line in enumerate(lines):
    # Fix 1: Scale intrinsics to match actual image resolution
    # After H, W = gt_image.shape[0], gt_image.shape[1], add intrinsic scaling
    if 'H, W = gt_image.shape[0], gt_image.shape[1]' in line:
        indent = line[:len(line) - len(line.lstrip())]
        # Add scaling code after this line
        scale_lines = [
            indent + '# Scale intrinsics if image resolution differs from camera calibration\n',
            indent + 'cam_w, cam_h = cam["width"], cam["height"]\n',
            indent + 'sx = W / cam_w\n',
            indent + 'sy = H / cam_h\n',
        ]
        for j, sl in enumerate(scale_lines):
            lines.insert(i + 1 + j, sl)
        changes += 1
        print(f'Fixed line {i+1}: added intrinsic scaling')
    
    # Fix 2: Apply scaling to K matrix
    if "cam['fx'], 0, cam['cx']" in line:
        lines[i] = line.replace("cam['fx']", "cam['fx'] * sx").replace("cam['cx']", "cam['cx'] * sx")
        changes += 1
        print(f'Fixed line {i+1}: scale fx, cx')
    if "cam['fy'], cam['cy']" in line and 'cam[\'fy\'] * sy' not in line:
        lines[i] = line.replace("cam['fy']", "cam['fy'] * sy").replace("cam['cy']", "cam['cy'] * sy")
        changes += 1
        print(f'Fixed line {i+1}: scale fy, cy')
    
    # Fix 3: Increase far_plane
    if 'far_plane=100.0' in line:
        lines[i] = line.replace('far_plane=100.0', 'far_plane=1000.0')
        changes += 1
        print(f'Fixed line {i+1}: far_plane 100 -> 1000')

with open(p, 'w') as f:
    f.writelines(lines)

print(f'\nTotal changes: {changes}')
