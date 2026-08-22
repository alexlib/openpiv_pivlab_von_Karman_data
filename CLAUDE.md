# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **tutorial**, not a library: a series of interactive [marimo](https://marimo.io) notebooks
teaching modern `openpiv-python` + [pivpy](https://github.com/alexlib/pivpy) usage, built around
a real dataset (a von Kármán vortex street behind a cylinder) with a PIVlab (MATLAB) analysis of
the same sequence to compare against. See `README.md` for the narrative and `FINDINGS.md` for
UX/bug findings surfaced in `openpiv-python`/`pivpy` while building this.

Image data from @Shrediquette (Google Drive). PIVlab data was produced independently in MATLAB
and is committed as-is (`PIVlab results/6pixels/`) — there's no way to regenerate it from this
repo, treat it as a fixed comparison baseline.

## Environment

Managed with `uv` (`pyproject.toml` + `uv.lock`).

```bash
uv sync                                        # install deps into .venv
uv run marimo edit notebooks/01_load_and_mask.py   # interactive
uv run notebooks/03_multipass_piv.py               # run as a script (non-interactive)
uv run marimo check notebooks/0*.py                # lint all notebooks
```

Notebooks also carry PEP 723 inline dependency headers, so `uv run notebooks/NN_*.py` works even
outside this project's venv (e.g. via `uvx marimo run`).

## Notebook structure and order

`notebooks/01` through `notebooks/05` are meant to be read/run **in order** — each depends on
concepts (and in the case of 03→04→05, on-disk output) from the previous one:

1. `01_load_and_mask.py` — masking the cylinder with `openpiv.preprocess.dynamic_masking`.
2. `02_choosing_parameters.py` — window size/overlap trade-offs, S/N threshold estimation, via
   single-pass `openpiv.pyprocess.extended_search_area_piv` calls (not the full pipeline).
3. `03_multipass_piv.py` — the full 4-pass windowed-deformation pipeline via
   `openpiv.windef.PIVSettings` + `windef.piv(settings)`, run across the whole `data/` sequence.
   Writes `notebooks/results/OpenPIV_results_6_tutorial/field_A%04d.txt` (gitignored — generated,
   not committed). **Takes a couple of minutes** (11 image pairs × 4-pass multipass PIV).
4. `04_vorticity_with_pivpy.py` — loads `03`'s output into pivpy's xarray `.piv` accessor and
   computes vorticity. **Does not use `pivpy.io.read_directory`/`read_piv`** for OpenPIV text
   output — it inverts the mask column (see `FINDINGS.md` #1) and silently returns all-zero
   velocity. Instead uses a small local loader (`load_openpiv_field`/`load_openpiv_sequence`,
   duplicated verbatim in `05` since each notebook must run standalone) that calls
   `pivpy.io.build_dataset` directly with the mask sense corrected.
5. `05_compare_to_pivlab.py` — loads the PIVlab session `.mat` file directly via `h5py` (not
   `pivpy.io.load_pivlab`, which expects a different/older PIVlab export convention — see
   `FINDINGS.md` #2), matches PIVlab's coordinate convention to OpenPIV's (`windef.piv()` applies
   `transform_coordinates`: flips `y` row order, negates `v`), and compares **`u, v` first**
   before vorticity — vorticity is a derivative and amplifies small misalignments between two
   independently-computed fields, so it's a misleading first comparison metric (see
   `FINDINGS.md`).

If you add a 6th notebook or modify shared logic (the loader functions), keep every notebook
independently runnable — marimo notebooks in this repo intentionally duplicate small helpers
rather than importing between them.

## Working with the PIV pipeline

- Current `openpiv-python` (v0.25.4) API differs from older tutorials you may find online:
  `windef.Settings` → `windef.PIVSettings`, `iterations` → `num_iterations`, and the old manual
  `first_pass()` + loop of `multipass_img_deform()` calls is now just `windef.piv(settings)`.
- `openpiv.validation.*` functions return a single boolean flag array (`True` = invalid); combine
  multiple validations with `|`, and pass the combined flag to `openpiv.filters.replace_outliers`.
- When changing PIV settings, keep them matched between `notebooks/03_multipass_piv.py` and
  `PIVlab results/6pixels/PIVlab_settings.txt` (window sizes, overlap, correlation quality) —
  mismatched parameters invalidate the comparison in notebook 5.
- `data/` and `PIVlab results/` are fixed inputs — don't regenerate or edit them.
