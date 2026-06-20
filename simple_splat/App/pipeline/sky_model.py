"""Learned sky / environment model — the Postshot "Create Sky Model" technique.

Instead of trying to reconstruct the sky with Gaussians (which spawns floaters,
spikes and halos), we fit a small MLP that maps a per-pixel ray DIRECTION to an
RGB colour. During training the rendered Gaussians (foreground) are composited
OVER this sky background using the Gaussian alpha:

    final = gaussian_rgb + (1 - gaussian_alpha) * sky(ray_dir)

The sky then "explains" all the textureless background, so the Gaussians get no
gradient to grow there -> clean foreground, coherent thin structures (flagpoles,
wires), and a perfect sky by construction. Trained jointly with the Gaussians.

The sky model is a training aid; for the viewer we still composite Brush's sky
Gaussians (which the user likes) — the sky model's job is to keep the FOREGROUND
clean during training.
"""
import torch
import torch.nn as nn


class SkyModel(nn.Module):
    def __init__(self, hidden=64, levels=4):
        super().__init__()
        self.levels = levels
        in_dim = 3 + 3 * 2 * levels          # raw dir + sin/cos positional encoding
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 3), nn.Sigmoid(),
        )

    def _encode(self, d):
        enc = [d]
        for i in range(self.levels):
            f = float(2 ** i)
            enc.append(torch.sin(f * d))
            enc.append(torch.cos(f * d))
        return torch.cat(enc, dim=-1)

    def forward(self, dirs):                  # dirs [..., 3] unit vectors
        return self.net(self._encode(dirs))   # [..., 3] in [0,1]


def ray_directions(K, c2w, H, W, device):
    """World-space unit ray direction per pixel. K [3,3], c2w [4,4]."""
    j, i = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    x = (i + 0.5 - K[0, 2]) / K[0, 0]
    y = (j + 0.5 - K[1, 2]) / K[1, 1]
    dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)      # [H,W,3]
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)
    R = c2w[:3, :3]
    dirs_world = dirs_cam @ R.T                                     # [H,W,3]
    return dirs_world
