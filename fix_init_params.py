import os

p = r'F:\Codebase\images_to_play\simple_splat\App\gsplat_mcmc_trainer.py'
with open(p, 'r') as f:
    c = f.read()

changes = []

# Fix 1: Lower initial opacity (0.1 instead of 0.5)
if 'opacities = torch.zeros(N)' in c:
    c = c.replace('opacities = torch.zeros(N)', 
                  'opacities = torch.full((N,), torch.logit(torch.tensor(0.1)).item())')
    changes.append('Opacity: logit(0.1)=-2.2 instead of 0 (was 50%, now 10%)')

# Fix 2: Random quaternions instead of identity
old_quat = """    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0"""
new_quat = "    quats = torch.rand(N, 4)"
if old_quat in c:
    c = c.replace(old_quat, new_quat)
    changes.append('Quaternions: random instead of identity')

# Fix 3: Per-point adaptive scale using knn
old_scale = """    # Use uniform scale from subsample estimate (fast, good enough for init)
    scales = torch.full((N, 3), global_avg_scale)"""
new_scale = """    # Per-point scale from nearest neighbor distances (matches gsplat official)
    tree_all = cKDTree(xyz)
    dists_all, _ = tree_all.query(xyz, k=min(4, N))
    dist_avg = np.sqrt((dists_all[:, 1:] ** 2).mean(axis=1))
    scales_np = np.log(dist_avg * 1.0).astype(np.float32)
    scales = torch.from_numpy(scales_np).unsqueeze(-1).repeat(1, 3)"""
if old_scale in c:
    c = c.replace(old_scale, new_scale)
    changes.append('Scale: per-point knn adaptive instead of uniform')

with open(p, 'w') as f:
    f.write(c)

print(f'Changes: {len(changes)}')
for ch in changes:
    print(f'  - {ch}')
