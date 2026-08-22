# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pivpy>=0.1.2",
#     "h5py",
#     "matplotlib>=3.11.1",
#     "numpy",
#     "xarray",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import h5py
    from pathlib import Path
    from pivpy import io
    import pivpy.pivpy  # noqa: F401 -- registers the .piv xarray accessor

    REPO_ROOT = Path(__file__).parent.parent
    OPENPIV_RESULTS_DIR = Path(__file__).parent / "results" / "OpenPIV_results_6_tutorial"
    PIVLAB_MAT = REPO_ROOT / "PIVlab results" / "6pixels" / "Karman_frame_00-10_intarea_6px_vorticity.mat"
    return OPENPIV_RESULTS_DIR, PIVLAB_MAT, h5py, io, mo, np, plt, xr


@app.cell
def _(mo):
    mo.md(
        r"""
        # 5. Comparing OpenPIV to PIVlab

        Both tools were run with matching settings (see
        `PIVlab results/6pixels/PIVlab_settings.txt` vs the `PIVSettings` in
        notebook 3): 4-pass window deformation, 64→32→16→6 px, 50% overlap,
        Gaussian subpixel, "Extreme"/normalized correlation quality. This
        notebook loads both tools' output for the same frame and compares them
        — on velocity first, then vorticity.
        """
    )
    return


@app.cell
def _(io, np, xr):
    def load_openpiv_field(path, frame):
        """Same OpenPIV-txt loader as notebook 4 -- see FINDINGS.md for why we
        don't use pivpy.io.read_directory directly (it inverts the mask)."""
        data = np.loadtxt(path, comments="#")
        cols = len(np.unique(data[:, 0]))
        rows = len(np.unique(data[:, 1]))
        x = data[:, 0].reshape(rows, cols)[0, :]
        y = data[:, 1].reshape(rows, cols)[:, 0]
        u = data[:, 2].reshape(rows, cols)
        v = data[:, 3].reshape(rows, cols)
        mask = data[:, 5].reshape(rows, cols)
        chc = 1.0 - mask
        return io.build_dataset(x, y, u, v, chc=chc, frame=frame, delta_t=1.0)


    def load_openpiv_sequence(results_dir):
        paths = sorted(results_dir.glob("field_A*.txt"))
        frames = [load_openpiv_field(p, i) for i, p in enumerate(paths)]
        return xr.concat(frames, dim="t")
    return (load_openpiv_sequence,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Loading PIVlab's `.mat` output

        The `.mat` file here is a **PIVlab session file** (MATLAB v7.3 / HDF5),
        not a per-frame export -- it stores the entire PIVlab UI state plus a
        `resultslist` cell array. Each row is one frame; the columns we need are
        `x, y, u, v` (columns 0-3). PIVlab also saved a `derived` cell array
        holding the scalar field it had selected for on-screen display when this
        session was saved (vorticity, per the filename) — we use that only for
        the qualitative plot below, not for the quantitative check (see why in
        the vorticity section).

        `pivpy.io.load_pivlab` expects a different, older PIVlab export
        convention (`u_<n>`/`v_<n>` variables) and raises `ValueError: No u_<n>
        datasets found` on this file -- another entry for `FINDINGS.md`. We read
        the session file directly with `h5py` instead.
        """
    )
    return


@app.cell
def _(h5py, np):
    def load_pivlab_frame(mat_path, frame):
        """Read one frame's x, y, u, v, and PIVlab's own derived scalar field
        from a PIVlab session .mat file."""
        with h5py.File(mat_path, "r") as f:
            resultslist = f["resultslist"]
            x = f[resultslist[frame, 0]][:][:, 0]
            y = f[resultslist[frame, 1]][:][0, :]
            u = f[resultslist[frame, 2]][:].T
            v = f[resultslist[frame, 3]][:].T
            derived = f[f["derived"][frame, 0]][:].T
        # PIVlab keeps the image convention (y downward, u/v in image axes).
        # OpenPIV's windef.piv() applies transform_coordinates: flips the row
        # order and negates v, to get a right-handed "physical" y-up frame.
        # Match that here so the two are directly comparable.
        return x, y[::-1], u[::-1, :], -v[::-1, :], derived[::-1, :]
    return (load_pivlab_frame,)


@app.cell
def _(mo):
    frame_picker = mo.ui.slider(start=0, stop=10, step=1, value=0, label="frame")
    frame_picker
    return (frame_picker,)


@app.cell
def _(OPENPIV_RESULTS_DIR, frame_picker, load_openpiv_sequence):
    openpiv_ds = load_openpiv_sequence(OPENPIV_RESULTS_DIR)
    openpiv_frame = openpiv_ds.isel(t=frame_picker.value)
    return (openpiv_frame,)


@app.cell
def _(PIVLAB_MAT, frame_picker, load_pivlab_frame, xr):
    pl_x, pl_y, pl_u, pl_v, pl_derived = load_pivlab_frame(PIVLAB_MAT, frame_picker.value)
    pivlab_frame = xr.Dataset(
        {
            "u": (("y", "x"), pl_u),
            "v": (("y", "x"), pl_v),
            "derived": (("y", "x"), pl_derived),
        },
        coords={"y": pl_y, "x": pl_x},
    )
    return (pivlab_frame,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Velocity comparison (the primitive, and the trustworthy check)

        Interpolate PIVlab's `u, v` onto OpenPIV's grid and compare directly —
        no derivatives involved yet, so this is the cleanest apples-to-apples
        check that both tools tracked the same flow.
        """
    )
    return


@app.cell
def _(np, openpiv_frame, pivlab_frame):
    pivlab_on_openpiv_grid = pivlab_frame.interp(
        x=openpiv_frame["x"], y=openpiv_frame["y"]
    )

    def rms_and_corr(a, b):
        valid = np.isfinite(a.values) & np.isfinite(b.values)
        diff = a.values[valid] - b.values[valid]
        rms = float(np.sqrt(np.mean(diff**2)))
        corr = float(np.corrcoef(a.values[valid], b.values[valid])[0, 1])
        return rms, corr

    u_rms, u_corr = rms_and_corr(openpiv_frame["u"], pivlab_on_openpiv_grid["u"])
    v_rms, v_corr = rms_and_corr(openpiv_frame["v"], pivlab_on_openpiv_grid["v"])
    {"u": {"rms": u_rms, "corr": u_corr}, "v": {"rms": v_rms, "corr": v_corr}}
    return (pivlab_on_openpiv_grid,)


@app.cell
def _(mo):
    mo.md(
        r"""
        `v` (the cross-stream component, spatially structured by the vortex
        shedding) typically shows a clear positive correlation — that confirms
        the coordinate handling above (flip + negate `v`) is correct: a
        negative or near-zero `v` correlation would mean the sign/orientation
        is still wrong, not that the tools disagree. `u` (the streamwise
        component) is dominated by a fairly uniform mean flow, so it has low
        *spatial variance* — correlation coefficient is a weak statistic for
        an almost-constant field, even when the two tools' RMS difference in
        `u` is small relative to its magnitude. Look at RMS relative to the
        field's own range, not just the correlation number.
        """
    )
    return


@app.cell
def _(frame_picker, openpiv_frame, pivlab_frame, plt):
    fig_quiver, (ax_ou, ax_pu) = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    step = 6  # subsample for a readable quiver plot

    ax_ou.quiver(
        openpiv_frame["x"][::step], openpiv_frame["y"][::step],
        openpiv_frame["u"].values[::step, ::step], openpiv_frame["v"].values[::step, ::step],
        scale=40,
    )
    ax_ou.set_title(f"OpenPIV, frame {frame_picker.value}")
    ax_ou.set_aspect(1)

    ax_pu.quiver(
        pivlab_frame["x"][::step], pivlab_frame["y"][::step],
        pivlab_frame["u"].values[::step, ::step], pivlab_frame["v"].values[::step, ::step],
        scale=40,
    )
    ax_pu.set_title(f"PIVlab, frame {frame_picker.value}")
    ax_pu.set_aspect(1)
    fig_quiver
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Vorticity comparison — and why it looks noisier than velocity

        Vorticity is a **spatial derivative** (`dv/dx - du/dy`), computed
        independently by each tool from its own (independently gridded,
        independently masked) velocity field. Differentiating amplifies any
        small misalignment between the two grids or subtle differences in how
        each tool filters/interpolates near the mask — so even when `u, v`
        agree well (as above), a pointwise vorticity correlation can come out
        much weaker. That's a property of comparing derivatives across
        independently-computed fields, not evidence the tools disagree on the
        underlying flow. **This is why we validated on `u, v` first**: judging
        agreement from the vorticity field alone would have been misleading.

        The plots below use pivpy's `.piv.vorticity()` for OpenPIV (recomputed
        from the `u, v` we already validated) and PIVlab's own saved `derived`
        field, both on the fixed **-0.3 to +0.3 (1/frame)** scale from the
        original repo, for a qualitative side-by-side.
        """
    )
    return


@app.cell
def _(frame_picker, openpiv_frame, pivlab_frame, plt):
    openpiv_w = openpiv_frame.piv.vorticity()["w"]

    fig_vort, (ax_ov, ax_pv) = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    ax_ov.pcolormesh(openpiv_w["x"], openpiv_w["y"], openpiv_w.values, cmap="jet", vmin=-0.3, vmax=0.3)
    ax_ov.set_title(f"OpenPIV vorticity, frame {frame_picker.value}")
    ax_ov.set_aspect(1)

    im_pv = ax_pv.pcolormesh(
        pivlab_frame["x"], pivlab_frame["y"], pivlab_frame["derived"].values,
        cmap="jet", vmin=-0.3, vmax=0.3,
    )
    ax_pv.set_title(f"PIVlab vorticity, frame {frame_picker.value}")
    ax_pv.set_aspect(1)

    plt.colorbar(im_pv, ax=[ax_ov, ax_pv], fraction=0.023, label="vorticity (1/frame)")
    fig_vort
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Takeaway

        With matched multipass settings, OpenPIV and PIVlab velocity fields
        correlate well once the coordinate/sign convention is matched — that's
        the real validation. Their vorticity fields show the same qualitative
        vortex-shedding pattern, but expect the pointwise agreement to be
        weaker than for `u, v`, since differentiation amplifies small grid and
        masking differences between two independent implementations.

        This closes the analysis loop: notebook 1 masked the cylinder,
        notebook 2 chose the window/overlap/threshold parameters, notebook 3
        ran the full multipass pipeline, notebook 4 computed vorticity with
        pivpy, and this notebook validated the result against an independent
        tool. See `FINDINGS.md` for the concrete openpiv-python/pivpy issues
        surfaced along the way.
        """
    )
    return


if __name__ == "__main__":
    app.run()
