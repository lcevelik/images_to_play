import os

p = r'F:\Codebase\images_to_play\simple_splat\App\gsplat_mcmc_trainer.py'
with open(p, 'r') as f:
    c = f.read()

# Fix 1: Correct the comment and invert w2c to c2w
old = """        # pycolmap's cam_from_world is world-to-camera (w2c), NOT camera-to-world
        # gsplat viewmats expect w2c as 4x4, so pad the 3x4 matrix
        w2c_3x4 = np.array(image.cam_from_world().matrix(), dtype=np.float32)
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :] = w2c_3x4
        # Keep on CPU — moved to GPU on-demand
        c2w_mats.append(torch.from_numpy(w2c_4x4))"""

new = """        # pycolmap's cam_from_world is world-to-camera (w2c)
        # gsplat viewmats expect camera-to-world (c2w), so invert it
        w2c_3x4 = np.array(image.cam_from_world().matrix(), dtype=np.float32)
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :] = w2c_3x4
        c2w_4x4 = np.linalg.inv(w2c_4x4)
        # Keep on CPU — moved to GPU on-demand
        c2w_mats.append(torch.from_numpy(c2w_4x4))"""

if old in c:
    c = c.replace(old, new)
    with open(p, 'w') as f:
        f.write(c)
    print('Fixed! w2c inverted to c2w')
else:
    print('Pattern not found')
    # Try to find the actual text
    for i, line in enumerate(c.split('\n'), 1):
        if 'cam_from_world' in line or 'w2c_3x4' in line:
            print(f'{i}: {repr(line)}')
