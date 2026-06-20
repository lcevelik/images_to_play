"""Generate sky masks for a folder of images using SegFormer (ADE20K).

In ADE20K, class id 2 is 'sky'. We segment each image and write a foreground
mask PNG (255 = keep / real surface, 0 = sky) next to it under <out>/masks/.
The MCMC trainer multiplies its loss by this mask so it never tries to
reconstruct the sky -> no sky floaters, and all capacity goes to real surfaces.
"""
import os
import glob
import numpy as np


def generate_sky_masks(image_dir, out_dir, model_name="nvidia/segformer-b1-finetuned-ade-512-512",
                       device="cuda", min_sky_frac=0.05, progress=print):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    proc = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(dev).eval()
    ADE_SKY = 2

    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    imgs = sorted([f for f in glob.glob(os.path.join(image_dir, '*')) if f.endswith(exts)])
    progress(f"[sky] {len(imgs)} images, model={model_name}, device={dev}")

    sky_fracs = []
    for i, f in enumerate(imgs):
        img = Image.open(f).convert("RGB")
        W, H = img.size
        inp = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**inp).logits                       # [1,150,h/4,w/4]
        up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        seg = up.argmax(1)[0].cpu().numpy()                    # [H,W]
        sky = (seg == ADE_SKY)
        fg = (~sky).astype(np.uint8) * 255                     # 255=keep, 0=sky
        name = os.path.splitext(os.path.basename(f))[0] + ".png"
        Image.fromarray(fg, mode="L").save(os.path.join(masks_dir, name))
        sky_fracs.append(float(sky.mean()))
        if (i + 1) % 10 == 0 or i == len(imgs) - 1:
            progress(f"[sky] {i+1}/{len(imgs)} (avg sky {np.mean(sky_fracs)*100:.1f}%)")

    mean_sky = float(np.mean(sky_fracs))
    progress(f"[sky] done | mean sky fraction {mean_sky*100:.1f}% "
             f"(min {np.min(sky_fracs)*100:.1f}%, max {np.max(sky_fracs)*100:.1f}%)")

    # Self-gating: if there's essentially no sky, this is an indoor / low-sky scene
    # (walls/ceilings are REAL surfaces we want to reconstruct, not mask). Remove the
    # masks so the trainer reconstructs everything normally — same pipeline, no sky
    # branch. Outdoor scenes keep their masks. Windows showing sky still get masked.
    if mean_sky < min_sky_frac:
        import shutil
        shutil.rmtree(masks_dir, ignore_errors=True)
        progress(f"[sky] mean sky {mean_sky*100:.1f}% < {min_sky_frac*100:.0f}% "
                 f"-> indoor/low-sky scene: masks removed, trainer reconstructs everything")
        return None
    progress(f"[sky] outdoor scene: masks kept -> {masks_dir}")
    return masks_dir


if __name__ == "__main__":
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--images", "-i", required=True)
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--model", default="nvidia/segformer-b1-finetuned-ade-512-512")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--min-sky-frac", type=float, default=0.05,
                   help="below this mean sky fraction the scene is treated as indoor (masks removed)")
    a = p.parse_args()
    t0 = time.time()
    generate_sky_masks(a.images, a.out, a.model, a.device, a.min_sky_frac,
                       progress=lambda m: print(f"[{int(time.time()-t0)}s] {m}", flush=True))
