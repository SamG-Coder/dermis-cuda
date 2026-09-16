# DERMIS — CUDA Human Laboratory

[Live demo](https://samg-coder.github.io/dermis-cuda/) · [Showcase recorder](https://samg-coder.github.io/dermis-cuda/showcase.html) · [Build & deployment](https://github.com/SamG-Coder/dermis-cuda/actions)

A real-time anatomical portrait with procedural skin, eyes and strand hair.
Edit the face, orbit the camera, change the lighting, enable living motion,
and push the renderer to 100,000 scalp strands.

**CUDA-authored, WebGPU-executed.** CUDA WebShader compiles the supported CUDA
source to WGSL in the browser. No native NVIDIA CUDA installation is required.
Use a WebGPU-enabled browser with hardware acceleration on HTTPS or localhost.
Performance and available memory depend on the device.

## Try it

Open the [live demo](https://samg-coder.github.io/dermis-cuda/) and initialize the
portrait. Drag to orbit, scroll to zoom, and right-drag to pan. The Face, Profile
and other camera shortcuts help inspect the model. Enable **Living portrait**
for blinking and subtle gaze changes. Use **View .cu** to inspect the source.

For the recorded showcase look: choose Extreme quality, skin pigment 0.00,
roughness 0.57, diffusion 0.55, detail 0.80, hazel/amber iris, pupil 0.33,
hair pigment 1.00, hair roughness 1.00, length 1.80, breeze 1.00, and living on.

## How it is built

1. **Anatomy:** a bundled MakeHuman CC0 mesh is cropped, normalized and subdivided.
   Authored closure targets provide eyelid motion. Source data and the rebuild
   script are included under `assets/source/`.
2. **Geometry:** `src/anatomy.js` prepares rest data, eyes, brows and individual
   scalp-conforming hair strands. `kernels/human.cu` deforms the face and fibers
   on the GPU. Geometry stays on the GPU during ordinary rendering.
3. **Materials:** `kernels/realism.cu` evaluates pore normals, pigment variation,
   regional roughness, iris fibers, crypts and sclera variation. Fine pore and
   iris detail is filtered using the pixel footprint to reduce shimmer.
4. **Lighting:** `kernels/appearance.cu` shades the portrait, approximates skin
   diffusion, adds directional strand highlights and accumulates samples.
5. **Presentation:** a small WebGPU raster bridge projects the geometry into
   position and normal buffers. CUDA-authored compute evaluates the final colour.

There are no photographic skin/eye textures, neural rendering models or Epic
MetaHuman assets. This is an experimental procedural renderer, not a scan or a
claim of photorealism. Skin diffusion is an artistic screen-space approximation;
the eyes do not implement physical corneal refraction.

## Quality and stress testing

| Preset | Scalp strands | Moving-frame samples | Longest render side cap |
| --- | ---: | ---: | ---: |
| Preview | 4,500 | 2 | 640 |
| Balanced | 12,000 | 4 | 1,100 |
| High | 22,000 | 8 | 1,600 |
| Stress | 50,000 | 8 | 2,000 |
| Extreme | 100,000 | 8 | 2,000 |

Stationary views converge over 24 samples. Moving frames sample the same pose
several times, without blending previous poses. Toggling living motion preserves
the quality preset, geometry and resolution. Head topology stays constant across
the presets; hair geometry changes. The UI reports actual triangle counts.

The timed stress test runs for 15 seconds, 60 seconds or 5 minutes at the selected
quality. It rebuilds geometry, shadows and shading, waits for GPU completion, and
exports a JSON report. Stop restores the authored look. Hiding the tab stops a run.
FPS includes browser frame scheduling; render milliseconds include CPU submission
and GPU queue completion. Buffer MiB excludes textures and is not total VRAM usage.

## Run locally

Node.js 20 or later is sufficient; there are no package installation steps.

```sh
npm start
# open http://localhost:8080
```

```sh
npm test
npm run compile
npm run test:server
npm run build
```

`dist/` is the static application. Serve it on localhost or HTTPS, not `file://`.
GitHub Actions runs tests, compilation and the server checks before deploying
`dist/` to GitHub Pages. Pull requests validate without deployment. Hosted CI
does not claim hardware GPU testing; the optional browser GPU scripts require
Playwright and a WebGPU-capable browser on the test host.

Compile source in this order: `human.cu`, `realism.cu`, `appearance.cu`. The app
and CLI assemble the source automatically. Programmatic `PortraitEngine` callers
pass realism followed by appearance as the appearance source. Legacy procedural
head kernels remain available when callers omit the anatomical asset.

## Live showcase

`showcase.html` records a 120-second, captioned, 1920 × 1080 tour directly from
the live renderer at Extreme quality. It demonstrates anatomy, maximum hair
length and breeze, blinking, skin, eyes, expression, diagnostic views and the
pipeline. Camera motion is scripted; rendering runs at wall-clock speed. It is
not an offline frame render or interpolated footage. The displayed FPS includes
capture overhead. The recording downloads locally as WebM; no upload occurs.

## Licenses

Application and bundled CUDA WebShader compiler/runtime: MIT. Anatomical graphical
assets: MakeHuman CC0 1.0 Universal. See [third-party notices](THIRD_PARTY_NOTICES.md)
and the [asset license](assets/MAKEHUMAN-CC0.md). No MakeHuman application code or
reference photographs are included. These assets are not restricted to Unreal Engine.
