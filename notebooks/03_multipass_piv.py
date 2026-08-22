# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "openpiv>=0.25.4",
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
    from openpiv import windef

    REPO_ROOT = Path(__file__).parent.parent
    DATA_DIR = REPO_ROOT / "data"
    RESULTS_DIR = Path(__file__).parent / "results"
    return DATA_DIR, RESULTS_DIR, mo, np, plt, windef


@app.cell
def _(mo):
    mo.md(
        r"""
        # 3. The full multipass windowed-deformation pipeline

        This reproduces the analysis from the original repo's
        `vonKarman_pivlab_with_openpiv.py`, matched to PIVlab's parameters for a
        fair comparison: 4 passes at window sizes **64 → 32 → 16 → 6 px**, 50%
        overlap, Gaussian subpixel fit, "Extreme" correlation quality (linear,
        normalized correlation).

        **API note for anyone who knows the old OpenPIV tutorials:** the 2021-era
        pipeline manually called `windef.first_pass()` then looped over
        `windef.multipass_img_deform()` — about 150 lines wiring passes,
        validation, outlier replacement, and smoothing together by hand. Modern
        OpenPIV (`windef.PIVSettings` + `windef.piv(settings)`) does all of that
        in one call: build a settings object, call `windef.piv`, get vector
        fields on disk. The knobs are still all there — they've just moved from
        scattered local variables into one typed settings object.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Build the settings

        These mirror `PIVlab_settings.txt` for the "6 pixels" case: same window
        sizes, same overlap, same masking region (cylinder only). We picked the
        masking threshold in notebook 1 and the window/overlap progression in
        notebook 2 — this is where they come together.
        """
    )
    return


@app.cell
def _(DATA_DIR, RESULTS_DIR, windef):
    settings = windef.PIVSettings()

    settings.filepath_images = DATA_DIR
    settings.save_path = RESULTS_DIR
    settings.save_folder_suffix = "tutorial"
    settings.frame_pattern_a = "karman_16Hz_*_A.jpg"
    settings.frame_pattern_b = "karman_16Hz_*_B.jpg"
    settings.roi = "full"

    # multipass window deformation: coarse -> fine, 50% overlap throughout
    settings.windowsizes = (64, 32, 16, 6)
    settings.overlap = (32, 16, 8, 3)
    settings.num_iterations = 4
    settings.subpixel_method = "gaussian"
    settings.deformation_method = "symmetric"
    settings.interpolation_order = 1

    # "Extreme" correlation quality
    settings.correlation_method = "linear"
    settings.normalized_correlation = True

    # mask the cylinder (not the shadow) -- see notebook 1
    settings.dynamic_masking_method = "intensity"
    settings.dynamic_masking_threshold = 0.1
    settings.dynamic_masking_filter_size = 21

    # validation -- see notebook 2 for how the S/N threshold is chosen
    settings.extract_sig2noise = True
    settings.sig2noise_method = "peak2peak"
    settings.sig2noise_mask = 2
    settings.validation_first_pass = True
    settings.min_max_u_disp = (-5, 5)
    settings.min_max_v_disp = (-5, 5)
    settings.std_threshold = 3
    settings.median_threshold = 3
    settings.median_size = 1
    settings.replace_vectors = True
    settings.filter_method = "localmean"
    settings.max_filter_iteration = 10
    settings.filter_kernel_size = 3

    # smooth between passes, not on the last one (windef.piv handles this)
    settings.smoothn = True
    settings.smoothn_p = 0.5

    settings.scaling_factor = 1
    settings.dt = 1
    settings
    return (settings,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Run it

        `windef.piv(settings)` scans `filepath_images` for every pair matching
        `frame_pattern_a`/`frame_pattern_b` (here all 11 `karman_16Hz_NNN_{A,B}.jpg`
        pairs), runs the full pipeline per pair, and writes `field_A%04d.txt`
        (x, y, u, v, validation flags, mask) into
        `save_path/OpenPIV_results_<last window size>_<suffix>/`. This processes
        the whole sequence, so it takes a couple of minutes.
        """
    )
    return


@app.cell
def _(settings, windef):
    windef.piv(settings)
    return


@app.cell
def _(RESULTS_DIR, settings):
    output_dir = (
        RESULTS_DIR
        / f"OpenPIV_results_{settings.windowsizes[settings.num_iterations - 1]}_{settings.save_folder_suffix}"
    )
    field_path = output_dir / "field_A0000.txt"
    field_path
    return (field_path,)


@app.cell
def _(field_path, np, plt):
    data = np.loadtxt(field_path)
    fx, fy, fu, fv = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    fig_result, ax_result = plt.subplots(figsize=(10, 5))
    ax_result.quiver(fx, fy, fu, fv, np.hypot(fu, fv), scale=30)
    ax_result.set_aspect(1)
    ax_result.set_title("final multipass field (physical y-up, x-right convention)")
    fig_result
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Takeaway

        `windef.piv` writes one `field_A%04d.txt` per image pair it finds,
        already validated and outlier-replaced, in physical (y-up) coordinates.
        That file format — plain-text `x y u v flag mask` columns — is exactly
        what pivpy's OpenPIV reader expects.

        Next: **04_vorticity_with_pivpy.py** — load these results into pivpy and
        compute vorticity.
        """
    )
    return


if __name__ == "__main__":
    app.run()
