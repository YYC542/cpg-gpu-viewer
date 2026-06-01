# CPG-GPU-Viewer

**GPU-accelerated 3D visualization of corner-point grid (CPG) reservoir models using an OpenGL geometry-shader pipeline, with virtual reality support.**

This repository contains the reference implementation accompanying the paper *"GPU-Accelerated 3D Visualization of Corner-Point Grid Reservoir Models with Geometry Shader Pipeline and Virtual Reality Integration."*

The engine reconstructs hexahedral reservoir cells entirely on the GPU from a compact per-cell corner representation, using a point-to-hexahedron expansion in an OpenGL 3.3 geometry shader. It is validated on the public **Norne ATW2013** benchmark (46×112×22 corner-point grid) and runs interactively on consumer hardware.

---

## Main program

> **The version used for all results reported in the paper is [`Nore viewer geometry shader v21.py`](./Nore%20viewer%20geometry%20shader%20v21.py).**

This is the recommended entry point. The other `Nore/Norne viewer geometry shader v*.py` files (v7–v23) are earlier development iterations kept for reference; they are **not** required to run the engine.

---

## Features

- **GPU-native CPG geometry generation** — each cell's eight corner points are stored in a texture buffer object (TBO); the geometry shader expands them into a hexahedron on the fly (single `glDrawArrays(GL_POINTS)` call per frame).
- **Depth-based fault screening** — an auxiliary heuristic flags large vertical ZCORN depth discontinuities (default threshold 30 m) as a visual screening aid (toggle with `F`).
- **Well trajectory overlays** — eight Norne wells rendered as 3D polylines (toggle with `W`).
- **Interactive cross-section slicing** — independent X and Z cutting planes.
- **Adjustable vertical exaggeration** — default 4×.
- **Rainbow depth colormap** — blue (shallow) → red (deep), consistent with common reservoir-visualization conventions.
- **Virtual reality** — stereoscopic dual-eye rendering via OpenVR / SteamVR (e.g. Windows Mixed Reality headsets), with first-frame pose anchoring and an `R`-key re-anchor mechanism.

---

## Requirements

- Python 3.8+
- A GPU supporting OpenGL 3.3 Core Profile (geometry shaders)
- Python packages:

```bash
pip install glfw PyOpenGL PyOpenGL-accelerate numpy
```

Virtual reality is **optional**. To use it, additionally install:

```bash
pip install openvr
```

and have **SteamVR** running with a connected headset. If `openvr` is not installed or no headset is detected, the program automatically falls back to desktop mode.

---

## Dataset

The Norne ATW2013 corner-point grid (`NORNE_ATW2013.GRDECL`) is included in this repository for convenience. It originates from the open **Open Porous Media (OPM)** initiative:

- https://github.com/OPM/opm-tests/tree/master/norne

Place the `.GRDECL` file in the same directory as the script (the default filename expected by the program is `NORNE_ATW2013.GRDECL`).

---

## Running

```bash
python "Nore viewer geometry shader v21.py"
```

The program loads the grid, reconstructs the geometry, and opens an interactive window. By default it attempts to enter VR mode and automatically falls back to desktop mode if no headset is available.

### Controls

| Key / Input | Action |
|---|---|
| Mouse drag | Rotate camera |
| Scroll | Zoom |
| `Q` / `A` | X cross-section (decrease / increase) |
| `E` / `D` | Z depth slice (decrease / increase) |
| `Z` / `X` | Vertical exaggeration (decrease / increase, default 4×) |
| `F` | Toggle fault-zone highlight (white tint) |
| `W` | Toggle well trajectories |
| `V` | Toggle VR / desktop mode |
| `R` | Re-anchor VR view (recenter if misaligned) |
| `ESC` | Quit |

**Colormap:** blue = shallow, red = deep. **White tint** marks flagged fault zones.

---

## Benchmarking

`norne_benchmark.py` and `run_all_benchmarks.py` provide a reproducible performance comparison between the geometry-shader pipeline, a CPU-side pre-generated VBO/VAO baseline, and legacy immediate mode, all using an identical camera path and timing harness.

```bash
python run_all_benchmarks.py
```

---

## License

Released under the [MIT License](./LICENSE).
