# Capture Tips (for the user guide)

Preserved from the Create-tab UI (removed 2026-06-22 to declutter; reuse when building the user guide).

Good capture is the single biggest factor in splat quality. Aim for **20–100 photos with 60–80% overlap**.

| Tip | Why it matters |
|-----|----------------|
| **Orbit the subject** | Move *around* the subject in a circle (or arc), keeping it framed — don't just pan from one spot. SfM needs parallax (the subject seen from different angles) to triangulate 3D points. |
| **60–80% overlap** | Each photo should share 60–80% of its view with the next one. Too little overlap and the reconstruction can't connect the images; too much wastes capture without adding detail. |
| **Even lighting** | Avoid harsh shadows, blown highlights, and changing light between shots. Consistent, diffuse light gives the matcher stable features and the splat consistent color. |
| **Texture over gloss** | Textured, matte surfaces reconstruct well; flat/glossy/reflective/transparent surfaces (asphalt, glass, water, white walls) give few SfM features and need extra tricks (learned matchers, depth seeding, the MCMC+Brush combine). |

## Quick checklist
- 20–100 photos (more for larger scenes)
- Circle the subject, keep it in frame
- 60–80% overlap between consecutive shots
- Even, consistent lighting (no flicker, no hard shadows)
- Prefer textured/matte surfaces; expect trouble on glossy/textureless ones
