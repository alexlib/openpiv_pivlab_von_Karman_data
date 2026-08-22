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
    import contextlib
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from openpiv import windef

    REPO_ROOT = Path(__file__).parent.parent
    DATA_DIR = REPO_ROOT / "data"
    RESULTS_DIR = Path(__file__).parent / "results"
    return DATA_DIR, RESULTS_DIR, contextlib, mo, np, plt, windef


@app.cell
def _(mo):
    mo.md(
        r"""
        # 3. The full multipass windowed-deformation pipeline

        This reproduces the analysis from the original repo's
        `vonKarman_pivlab_with_openpiv.py`, matched to PIVlab's parameters for a
        fair comparison: 4 passes at window sizes **64 → 32 → 16 → 6 px**, 50%
        overlap, Gaussian subpixel fit, "Extreme" correlation quality (linear,
        normalized correlation) — the defaults below reproduce that. The widgets
        let you change the settings that matter most for the result (window
        progression, correlation quality, masking, smoothing) and see the effect,
        the same knobs discussed in notebooks 1 and 2.

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
        ## Build the settings, interactively

        `PIVSettings` has ~40 fields; exposing all of them would be a form, not a
        lesson. These are the ones that change the result the most:
        """
    )
    return


@app.cell
def _(mo):
    window_progression_picker = mo.ui.dropdown(
        options={
            "64 -> 32 -> 16 -> 6  (this tutorial's default, 4 passes)": ((64, 32, 16, 6), (32, 16, 8, 3)),
            "64 -> 32 -> 16  (3 passes, coarser final grid)": ((64, 32, 16), (32, 16, 8)),
            "128 -> 64 -> 32 -> 16  (4 passes, bigger/more robust windows)": ((128, 64, 32, 16), (64, 32, 16, 8)),
            "32 -> 16  (2 passes, fast to iterate with)": ((32, 16), (16, 8)),
        },
        value="64 -> 32 -> 16 -> 6  (this tutorial's default, 4 passes)",
        label="window/overlap progression",
    )

    correlation_quality_picker = mo.ui.dropdown(
        options={
            "Extreme (linear, normalized)": ("linear", True),
            "Fast (circular, not normalized)": ("circular", False),
        },
        value="Extreme (linear, normalized)",
        label="correlation quality",
    )

    sequence_picker = mo.ui.dropdown(
        options={
            "single pair (000) -- fast, good for tuning": ("karman_16Hz_000_A.jpg", "karman_16Hz_000_B.jpg"),
            "full sequence (11 pairs) -- ~2 minutes": ("karman_16Hz_*_A.jpg", "karman_16Hz_*_B.jpg"),
        },
        value="full sequence (11 pairs) -- ~2 minutes",
        label="image pairs to process",
    )

    mo.hstack([window_progression_picker, correlation_quality_picker, sequence_picker])
    return correlation_quality_picker, sequence_picker, window_progression_picker


@app.cell
def _(mo):
    mask_threshold_slider = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.1, label="mask threshold"
    )
    mask_filter_slider = mo.ui.slider(
        start=3, stop=41, step=2, value=21, label="mask filter_size"
    )
    smoothn_checkbox = mo.ui.checkbox(value=True, label="smooth between passes")
    smoothn_p_slider = mo.ui.slider(
        start=0.01, stop=2.0, step=0.01, value=0.5, label="smoothn_p"
    )

    mo.hstack(
        [mask_threshold_slider, mask_filter_slider, smoothn_checkbox, smoothn_p_slider]
    )
    return mask_filter_slider, mask_threshold_slider, smoothn_checkbox, smoothn_p_slider


@app.cell
def _(mo):
    mo.md(
        r"""
        The rest (validation thresholds, filter method, subpixel method, ...) are
        kept fixed at PIVlab-matched defaults below — see `PIVSettings` in the
        cell that builds `settings` if you want to change those too.
        """
    )
    return


@app.cell
def _(
    DATA_DIR,
    RESULTS_DIR,
    correlation_quality_picker,
    mask_filter_slider,
    mask_threshold_slider,
    sequence_picker,
    smoothn_checkbox,
    smoothn_p_slider,
    window_progression_picker,
    windef,
):
    settings = windef.PIVSettings()

    settings.filepath_images = DATA_DIR
    settings.save_path = RESULTS_DIR
    settings.save_folder_suffix = "tutorial"
    settings.frame_pattern_a, settings.frame_pattern_b = sequence_picker.value
    settings.roi = "full"

    # multipass window deformation: coarse -> fine
    settings.windowsizes, settings.overlap = window_progression_picker.value
    settings.num_iterations = len(settings.windowsizes)
    settings.subpixel_method = "gaussian"
    settings.deformation_method = "symmetric"
    settings.interpolation_order = 1

    settings.correlation_method, settings.normalized_correlation = correlation_quality_picker.value

    # mask the cylinder (not the shadow) -- see notebook 1
    settings.dynamic_masking_method = "intensity"
    settings.dynamic_masking_threshold = mask_threshold_slider.value
    settings.dynamic_masking_filter_size = mask_filter_slider.value

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
    settings.smoothn = smoothn_checkbox.value
    settings.smoothn_p = smoothn_p_slider.value

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
        `frame_pattern_a`/`frame_pattern_b`, runs the full pipeline per pair, and
        writes `field_A%04d.txt` (x, y, u, v, validation flags, mask) into
        `save_path/OpenPIV_results_<last window size>_<suffix>/`. Defaults to the
        full 11-pair sequence (~2 minutes) so notebooks 4-5 have what they need;
        switch "image pairs to process" to "single pair" above while you're still
        tuning settings, since this run is gated behind a button rather than
        re-running on every widget tweak.

        `windef.piv` doesn't report progress on its own -- it's one opaque call by
        design (that's the point of the modern API). To see which phase is
        running (which pass, which window size, smoothing) without reimplementing
        the multipass loop ourselves, we temporarily wrap the internal functions
        it calls (`first_pass`, `multipass_img_deform`, `smoothn.smoothn`) to
        report to a progress bar, then put them back.
        """
    )
    return


@app.cell
def _(contextlib, windef):
    @contextlib.contextmanager
    def report_piv_progress(settings, bar):
        """Temporarily wrap windef's internal per-pass functions so a marimo
        progress bar shows which phase is running, without touching how
        windef.piv() itself works."""
        original_first_pass = windef.first_pass
        original_multipass_img_deform = windef.multipass_img_deform
        original_smoothn = windef.smoothn.smoothn
        pair_count = {"n": 0}

        def first_pass_wrapped(frame_a, frame_b, s):
            pair_count["n"] += 1
            ws = s.windowsizes[0]
            bar.update(
                increment=1,
                subtitle=f"pair {pair_count['n']}: pass 1/{s.num_iterations} ({ws}x{ws} px)",
            )
            return original_first_pass(frame_a, frame_b, s)

        def multipass_img_deform_wrapped(frame_a, frame_b, i, x, y, u, v, s):
            ws = s.windowsizes[i]
            bar.update(
                increment=1,
                subtitle=f"pair {pair_count['n']}: pass {i + 1}/{s.num_iterations} ({ws}x{ws} px)",
            )
            return original_multipass_img_deform(frame_a, frame_b, i, x, y, u, v, s)

        def smoothn_wrapped(*args, **kwargs):
            bar.update(increment=1, subtitle=f"pair {pair_count['n']}: smoothing")
            return original_smoothn(*args, **kwargs)

        windef.first_pass = first_pass_wrapped
        windef.multipass_img_deform = multipass_img_deform_wrapped
        windef.smoothn.smoothn = smoothn_wrapped
        try:
            yield
        finally:
            windef.first_pass = original_first_pass
            windef.multipass_img_deform = original_multipass_img_deform
            windef.smoothn.smoothn = original_smoothn
    return (report_piv_progress,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run PIV pipeline")
    run_button
    return (run_button,)


@app.cell
def _(DATA_DIR, mo, report_piv_progress, run_button, settings, windef):
    is_script_mode = mo.app_meta().mode == "script"
    mo.stop(
        not (is_script_mode or run_button.value),
        mo.md("*Click **Run PIV pipeline** above to execute with the current settings.*"),
    )

    num_pairs = len(list(DATA_DIR.glob(settings.frame_pattern_a)))
    # smoothn.smoothn is called twice per smoothing stage (once for u, once
    # for v), and smoothing happens after every pass except the last.
    smoothing_steps_per_pair = 2 * (settings.num_iterations - 1) if settings.smoothn else 0
    steps_per_pair = settings.num_iterations + smoothing_steps_per_pair
    total_steps = num_pairs * steps_per_pair

    with mo.status.progress_bar(
        total=total_steps,
        title="Running PIV",
        subtitle="starting...",
        completion_title="Done",
        completion_subtitle=f"{num_pairs} pair(s) processed",
    ) as progress_bar:
        with report_piv_progress(settings, progress_bar):
            windef.piv(settings)
    return


@app.cell
def _(RESULTS_DIR, settings):
    output_dir = (
        RESULTS_DIR
        / f"OpenPIV_results_{settings.windowsizes[settings.num_iterations - 1]}_{settings.save_folder_suffix}"
    )
    field_path = sorted(output_dir.glob("field_A*.txt"))[0]
    field_path
    return (field_path,)


@app.cell
def _(field_path, np, plt):
    data = np.loadtxt(field_path)
    fx, fy, fu, fv = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    fig_result, ax_result = plt.subplots(figsize=(10, 5))
    ax_result.quiver(fx, fy, fu, fv, np.hypot(fu, fv), scale=30)
    ax_result.set_aspect(1)
    ax_result.set_title("first pair's field (physical y-up, x-right convention)")
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
        what pivpy's OpenPIV reader expects (modulo the mask-sense bug noted in
        `FINDINGS.md`).

        Notebooks 4 and 5 expect the **full sequence** run
        (`OpenPIV_results_6_tutorial/`, the default window progression) to exist
        in `notebooks/results/` — if you switched to "single pair" or a different
        window progression while exploring above, switch back to the defaults
        (full sequence, 64→32→16→6) and click **Run PIV pipeline** once more
        before moving on.

        Next: **04_vorticity_with_pivpy.py** — load these results into pivpy and
        compute vorticity.
        """
    )
    return


if __name__ == "__main__":
    app.run()
