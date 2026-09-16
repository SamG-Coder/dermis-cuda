# Third-party notices

## CUDA WebShader

The compiler/runtime under `vendor/cuda-webshader` is from SamG-Coder's
CUDA WebShader project: https://github.com/SamG-Coder/cuda-webshader

Copyright (c) 2026 SamG-Coder and contributors. MIT license; the full license is
retained at `vendor/cuda-webshader/LICENSE`.

This project vendors the tested compiler/runtime snapshot from the previously
packaged CUDA Studio project, not a claim to track the repository's latest head.
The included file hashes identify the exact distributed snapshot.

## Original implementation

The DERMIS application, deformation and appearance kernels, strand construction
and tests are original source. The anatomical rest mesh is derived from the CC0
MakeHuman assets listed below. No skin/eye/hair textures, portrait photographs,
HDR images or pretrained weights are used.
The interface uses system fonts.

Research references in README.md are technical background, not copied shader
implementations or redistributable image/model assets. Dawn, SwiftShader, Chromium
and Playwright were used on the validation host; no native binary from those
projects is included in the application ZIP.

## MakeHuman anatomical assets (CC0 1.0 Universal)

`assets/male-anatomy.json` is derived from MakeHuman hm08 `base.obj`,
`caucasian-male-young.target`, and
`universal-male-young-averagemuscle-averageweight.target`.
The left and right eye-closure targets from expression/units/caucasian are also included.
Only graphical assets were used; no MakeHuman program source is included.
The mesh is cropped to the upper body, normalized to portrait coordinates and
Catmull-Clark subdivided twice by the original build script included with the source assets.

Source: https://github.com/makehumancommunity/makehuman
License: https://github.com/makehumancommunity/makehuman/blob/master/LICENSE.ASSETS.md
License copy: `assets/MAKEHUMAN-CC0.md`.
Source files, hashes and rebuild script: `assets/source/`.
