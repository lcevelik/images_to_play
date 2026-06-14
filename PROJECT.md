# Project: images_to_play (FonixFlow Splat)

<!-- 
Project Tracker — keep these H2 headings exactly as-is.
The tracker parses them to populate Kanban columns.
Edit the tasks below. Use - [ ] for open, - [x] for done.
-->

## Goals

- [ ] Best-possible Gaussian splats from casual photo/video captures, fully local (target: 2026-07-01)
- [ ] Robust pipeline: no capture should hard-fail without a usable fallback output

## In Progress

- (none — pick next item from To Do)

## To Do

- [ ] **Verify the 3D alignment preview end-to-end** (restart server + run a job; confirm SuperSplat renders the degree-0 PLY + cyan frustums look right)
- [ ] **Preview Phase 3** — slider to scrub training checkpoints (`export_<iter>.ply`) and watch the splat densify from the sparse cloud
- [ ] **Run HQ COLMAP on the 74 photos**, compare sparse-point count vs 31,373 (old) and RealityScan 53,612 — did #1–4 close the gap?
- [ ] **`rs_to_colmap.py` converter** (transforms.json → pinhole COLMAP + undistorted images) so MCMC can train RealityScan poses → then the 4-way A/B (brush/mcmc × colmap/realityscan, named clearly)
- [ ] **Learning-based matcher** (LightGlue / MASt3R-SfM / VGGT) as a COLMAP front-end — the real gap-closer to RealityScan on textureless surfaces
- [ ] **MCMC ADC mode** (gsplat `DefaultStrategy`, INRIA-style adaptive densification) — may look cleaner than MCMC for this content
- [ ] Expose `mcmc_cap` in Settings UI (auto by default; RTX 8000 handles 2–4M — biggest quality lever for detailed scenes)
- [ ] Head-to-head benchmark: Brush vs MCMC on the same 74-photo capture, same steps, PSNR on held views (samples uploaded to gallery)
- [ ] gsplat quality flags: `rasterize_mode="antialiased"` (check SuperSplat compat) and bilateral grid for auto-exposure captures
- [ ] VGGT / MASt3R-SfM fallback when COLMAP registration fails (write COLMAP-format output, continue pipeline)
- [ ] Optional: rename `simple_splat/` directory to match FonixFlow branding (touches CLAUDE.md, Brush path, scripts)
- [ ] Mesh export from splats (KIRI-style) — competitive gap noted in 2026-06-12 ecosystem review
- [ ] SPZ/SOG compressed splat export for web delivery

## Done

- [x] 2026-06-14 — 3D alignment preview in Process tab: `sparse_preview.py` writes sparse points + camera frustums as tiny Gaussians (pure-binary COLMAP parse, **no pycolmap** — bundled Python lacks it) → embedded SuperSplat, replaces the log panel. `/preview/<job_id>.ply` route + `preview_ready` status flag
- [x] 2026-06-14 — COLMAP HQ flags for high/quality/expert (`hq` gate): affine-shape SIFT + domain-size pooling, raised max_image_size, guided matching, BA refine principal point
- [x] 2026-06-14 — MCMC verified on real hardware (Quadro RTX 8000 48 GB, torch 2.12+cu126 at C:\Program Files\Python314); auto-cap confirmed (4M overfits a 74-photo scene → haze)
- [x] 2026-06-14 — Brush v0.3.0 progress/ETA from numbered `export_{iter}.ply` (GUI binary, no stdout; `--with-viewer false` rejected)
- [x] 2026-06-14 — RealityScan A/B: aligned same 74 photos 74/74, denser sparse (53,612 vs COLMAP 31,373); exported COLMAP (FULL_OPENCV) + Radiance Fields (`rs_nerf/transforms.json`, Brush-usable)
- [x] 2026-06-13 — Full/MCMC packaging: `build_lite_package.py --with-mcmc` (bundles torch+gsplat, NVIDIA-only) + `START_SERVER_MCMC.bat` (runs app under system CUDA Python)
- [x] 2026-06-13 — Brush v0.3.0 hardening: numbered `export_{iter}.ply` checkpoints, filesystem progress + ETA, timeout salvage, removed invalid `--with-viewer false`
- [x] 2026-06-13 — Desktop notification + beep on job completion
- [x] 2026-06-13 — Deep audit: 8 app.py bugs fixed (74→15 photo sampling, ZIP flatten, preset-downgrade sentinels, PLY validation) + `GlobalMapper.*` preset params wired (presets were producing identical reconstructions)
- [x] 2026-06-13 — Auto MCMC Gaussian cap scaled to sparse-point count (omit --cap-max = scene-aware, range 0.5M–2M)
- [x] 2026-06-13 — MCMC 4M Gaussians trained (1h37m RTX 8000), uploaded to splat.steadiczech.com as comparison sample
- [x] 2026-06-13 — Brush 50k-steps run completed and uploaded to splat.steadiczech.com for Brush vs MCMC comparison
- [x] 2026-06-12 — MCMC trainer fixed (viewmats w2c, optimizer wiring, SSIM, PLY export) and verified: 39 dB synthetic, 25.5 dB real capture
- [x] 2026-06-12 — Smoke test `test_mcmc_smoke.py` incl. viewer-style PLY reload check
- [x] 2026-06-12 — LPIPS on 512px crop (was full-res — 10× slower, VRAM risk)
- [x] 2026-06-12 — GUI redesign: tabbed layout (Create/Settings/Process/Results), no scroll, auto tab-switching
- [x] 2026-06-12 — Rebrand Simple Splat → FonixFlow Splat (UI, titles, docs)
- [x] 2026-06-12 — All .md docs refreshed to match code (presets, trainers, endpoints, modules)
- [x] 2026-06-12 — All work committed and pushed (commits 8cb840b…8280472)
- [x] 2026-06 — Camera tracking export UI (FBX/GLTF/JSON/Blender)
- [x] 2026-06 — Multi-GPU support for COLMAP + Brush, preset cleanup (5 presets), Brush LPIPS crash fix
- [x] 2026-05 — Phases 1–3: JSON presets, blur filter, auto-resize, quality scaler, Brush streaming, stage timings, MCMC trainer integration

## Blocked

- (none)

## Releases

- v0.1.0 — planned 2026-07-01 — First FonixFlow Splat release: verified dual-trainer pipeline + tabbed UI

## Notes

- Project created: 2026-05-26
- Working pipeline verified end-to-end 2026-06-12 on 74 iPhone photos (medium preset, MCMC trainer, ~20 min total)
- ROADMAP.md contains a detailed code review; its 2026-05-28 audit is partially stale — see the 2026-06-12 update at the top of that file
