# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pivpy>=0.1.2",
#     "matplotlib>=3.11.1",
#     "numpy",
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
    from pathlib import Path
    import xarray as xr
    from pivpy import io
    import pivpy.pivpy  # noqa: F401 -- registers the .piv xarray accessor

    RESULTS_DIR = (
        Path(__file__).parent / "results" / "OpenPIV_results_6_tutorial"
    )
    return RESULTS_DIR, io, mo, np, plt, xr


@app.cell
def _(mo):
    mo.md(
        r"""
        # 4. Vorticity with pivpy

        [pivpy](https://github.com/alexlib/pivpy) wraps PIV vector fields in
        `xarray`, and adds a `.piv` accessor with PIV-specific operations
        (vorticity, quiver/scalar plots, spatial averages, ...). The idiomatic
        way to load a directory of OpenPIV output is:

        ```python
        from pivpy import io
        import pivpy.pivpy  # registers the .piv accessor
        ds = io.read_directory(results_dir, pattern="field_A*", ext=".txt")
        ds = ds.piv.vorticity()
        ```

        **This doesn't work correctly here** — see the callout below — so this
        notebook loads the fields with a small local helper instead, and uses
        pivpy for everything downstream (`.piv.vorticity()`, plotting). That
        split — pivpy for xarray/analysis, hand-rolled I/O — is itself one of
        the "friction points" this tutorial project set out to find; see
        `FINDINGS.md`.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Bug found: `pivpy.io`'s OpenPIV text reader inverts the mask.**

            `openpiv.tools.save()` (what `windef.piv()` uses) writes a 6th
            column where **`mask == 1` means "invalid / masked out"** (e.g.
            inside the cylinder) and `mask == 0` means valid — that's the
            convention `openpiv.tools.save` and `openpiv.tools.display_vector_field`
            both use.

            `pivpy.io.OpenPIVReader.read` treats that column as if it were a
            *validity* flag instead (`invalid = mask == 0`), i.e. the opposite
            convention. On this dataset, only ~2.7% of grid points are inside
            the cylinder mask (`mask == 1`); the reader zeros out the other
            ~97% (`mask == 0`, the entire valid flow field) and keeps the
            near-zero cylinder points. The result: `ds['u']` and `ds['v']` come
            back **all zeros** for every frame, with no error or warning.

            Filed as a pivpy finding — see `FINDINGS.md` for the write-up and a
            reproduction.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(io, np, xr):
    def load_openpiv_field(path, frame):
        """Load one `field_A%04d.txt` (x y u v flags mask) with the correct
        OpenPIV mask convention: mask==1 is invalid, mask==0 is valid."""
        data = np.loadtxt(path, comments="#")
        cols = len(np.unique(data[:, 0]))
        rows = len(np.unique(data[:, 1]))
        x = data[:, 0].reshape(rows, cols)[0, :]
        y = data[:, 1].reshape(rows, cols)[:, 0]
        u = data[:, 2].reshape(rows, cols)
        v = data[:, 3].reshape(rows, cols)
        mask = data[:, 5].reshape(rows, cols)
        chc = 1.0 - mask  # pivpy's chc convention: 1 = valid, 0 = invalid
        return io.build_dataset(x, y, u, v, chc=chc, frame=frame, delta_t=1.0)


    def load_openpiv_sequence(results_dir):
        paths = sorted(results_dir.glob("field_A*.txt"))
        frames = [load_openpiv_field(p, i) for i, p in enumerate(paths)]
        return xr.concat(frames, dim="t")
    return (load_openpiv_sequence,)


@app.cell
def _(RESULTS_DIR, load_openpiv_sequence):
    ds = load_openpiv_sequence(RESULTS_DIR)
    ds
    return (ds,)


@app.cell
def _(ds, np):
    assert np.nanmax(np.abs(ds["u"].values)) > 0, "u is all zero -- loader regressed"
    return


@app.cell
def _(ds):
    vort = ds.piv.vorticity()
    vort
    return (vort,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Plot vorticity across the sequence

        The original repo's README used a fixed uncalibrated color scale of
        **-0.3 to +0.3 (1/frame)** for all frames, so successive frames are
        visually comparable. We do the same here.
        """
    )
    return


@app.cell
def _(mo, vort):
    frame_slider = mo.ui.slider(
        start=0, stop=int(vort.sizes["t"]) - 1, step=1, value=0, label="frame"
    )
    frame_slider
    return (frame_slider,)


@app.cell
def _(frame_slider, plt, vort):
    fig_vort, ax_vort = plt.subplots(figsize=(9, 4.5))
    w_frame = vort["w"].isel(t=frame_slider.value)
    im_vort = ax_vort.pcolormesh(
        w_frame["x"], w_frame["y"], w_frame.values, cmap="jet", vmin=-0.3, vmax=0.3
    )
    ax_vort.set_aspect(1)
    ax_vort.set_title(f"vorticity, frame {frame_slider.value}")
    plt.colorbar(im_vort, ax=ax_vort, fraction=0.046, label="1/frame")
    fig_vort
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Takeaway

        Once the field is correctly loaded, pivpy's `.piv` accessor makes
        vorticity a one-liner (`ds.piv.vorticity()`), and it's differentiated
        via `xarray`'s `.differentiate`, so it works the same whether `ds` has
        one frame or the whole sequence. That's the value pivpy adds over
        hand-rolling numpy gradients per frame — the I/O bug above is a rough
        edge on the way in, not a problem with pivpy's core `.piv` operations.

        Next: **05_compare_to_pivlab.py** — load the equivalent PIVlab result
        and compare.
        """
    )
    return


if __name__ == "__main__":
    app.run()
