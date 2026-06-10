import os

p = r'F:\Codebase\images_to_play\simple_splat\App\gsplat_mcmc_trainer.py'
with open(p, 'r') as f:
    c = f.read()

changes = []

# Fix 1: Revert c2w inversion (gsplat expects w2c)
if 'c2w_4x4 = np.linalg.inv(w2c_4x4)' in c:
    c = c.replace('        c2w_4x4 = np.linalg.inv(w2c_4x4)\n', '')
    changes.append('Removed c2w inversion')

if 'torch.from_numpy(c2w_4x4)' in c:
    c = c.replace('torch.from_numpy(c2w_4x4)', 'torch.from_numpy(w2c_4x4)')
    changes.append('Reverted to w2c_4x4')

# Fix 2: Fix SH initialization (remove -0.5 offset, gsplat uses standard SH)
if '(rgb_tensor - 0.5) / C0' in c:
    c = c.replace('(rgb_tensor - 0.5) / C0', 'rgb_tensor / C0')
    changes.append('Fixed SH init: rgb/C0 instead of (rgb-0.5)/C0')

# Fix 3: Update comment about viewmat convention
if 'gsplat viewmats expect camera-to-world (c2w), so invert it' in c:
    c = c.replace(
        'gsplat viewmats expect camera-to-world (c2w), so invert it',
        'gsplat viewmats expect world-to-camera (w2c)'
    )
    changes.append('Fixed comment')

with open(p, 'w') as f:
    f.write(c)

print(f'Changes: {len(changes)}')
for ch in changes:
    print(f'  - {ch}')
