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
    from openpiv import tools, pyprocess

    DATA_DIR = Path(__file__).parent.parent / "data"
    frame_a = tools.imread(DATA_DIR / "karman_16Hz_000_A.jpg")
    frame_b = tools.imread(DATA_DIR / "karman_16Hz_000_B.jpg")
    return frame_a, frame_b, mo, np, plt, pyprocess


@app.cell
def _(mo):
    mo.md(
        r"""
        # 2. Choosing PIV parameters

        A single PIV correlation pass has two core parameters:

        - **window size** — the side length (px) of the interrogation window
          correlated between frame A and frame B. Bigger windows contain more
          particle pairs, so the correlation peak is more reliable (higher
          signal-to-noise) — but the result is one averaged vector per window,
          so spatial resolution drops, and if the flow varies a lot inside one
          window, the vector is a blur of several different velocities.
        - **overlap** — how much neighboring windows overlap. 50% overlap (e.g.
          `overlap = window_size / 2`) is a good default: it doubles the vector
          density for free without changing what each vector measures.

        There's no "correct" window size in the abstract — it depends on
        particle density, displacement magnitude, and how fast the flow changes
        in space. Below, run a **single pass** at whatever settings you pick and
        see the result and its signal-to-noise map directly, on the same image
        pair used in notebook 1.
        """
    )
    return


@app.cell
def _(mo):
    window_size_picker = mo.ui.dropdown(
        options=["16", "32", "64", "128"], value="32", label="window size (px)"
    )
    overlap_fraction_slider = mo.ui.slider(
        start=0.0, stop=0.75, step=0.25, value=0.5, label="overlap fraction"
    )
    mo.hstack([window_size_picker, overlap_fraction_slider])
    return overlap_fraction_slider, window_size_picker


@app.cell
def _(overlap_fraction_slider, window_size_picker):
    window_size = int(window_size_picker.value)
    overlap = int(window_size * overlap_fraction_slider.value)
    return overlap, window_size


@app.cell
def _(frame_a, frame_b, overlap, pyprocess, window_size):
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        correlation_method="linear",
        normalized_correlation=True,
        sig2noise_method="peak2peak",
    )
    x, y = pyprocess.get_coordinates(frame_a.shape, window_size, overlap)
    return sig2noise, u, v, x, y


@app.cell
def _(sig2noise, u, v, x, y, plt):
    fig_single, (ax_quiver, ax_s2n) = plt.subplots(1, 2, figsize=(13, 4))
    ax_quiver.quiver(x, y, u, -v, scale=200)
    ax_quiver.invert_yaxis()
    ax_quiver.set_aspect(1)
    ax_quiver.set_title(f"single-pass field ({u.shape[1]}x{u.shape[0]} vectors)")

    im = ax_s2n.imshow(sig2noise, cmap="viridis")
    ax_s2n.set_title("signal-to-noise (peak2peak)")
    plt.colorbar(im, ax=ax_s2n, fraction=0.046)
    fig_single
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Try `window size = 16, overlap = 0`: the field gets visibly noisy —
        small windows near the resolution limit of the seeding density produce
        unreliable correlation peaks, seen as low/scattered S/N and spurious
        vectors. Try `window size = 64, overlap = 0.5`: much smoother, but the
        vortex street's finer structure gets smoothed away.

        This is exactly why the multipass **window deformation** approach (used
        in notebook 3) starts *coarse* (64 px, robust, low resolution) and
        refines *down* to a small final window (6 px here) using each pass's
        result to pre-deform the next pass's windows — you get the robustness of
        a big first guess and the resolution of a small final window, instead of
        having to pick one size and live with its trade-off.

        ## Setting the signal-to-noise threshold

        Rather than hand-picking a signal-to-noise cutoff, a common and robust
        approach is to look at the *distribution* of S/N values for a pass and
        take a low percentile (e.g. 5th) as the threshold — vectors below it are
        flagged as unreliable and replaced during validation.
        """
    )
    return


@app.cell
def _(np, sig2noise):
    valid_s2n = sig2noise[sig2noise > 0]
    threshold_5th_pct = np.percentile(valid_s2n, 5)
    threshold_5th_pct
    return threshold_5th_pct, valid_s2n


@app.cell
def _(threshold_5th_pct, valid_s2n, plt):
    fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
    ax_hist.hist(valid_s2n, bins=40)
    ax_hist.axvline(threshold_5th_pct, color="red", label=f"5th pct = {threshold_5th_pct:.2f}")
    ax_hist.set_xlabel("signal-to-noise ratio")
    ax_hist.legend()
    fig_hist
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Takeaway

        - Window size trades robustness (bigger) against resolution (smaller).
        - 50% overlap is a good default: more vectors, no cost.
        - The S/N threshold isn't a magic constant — estimate it per-run from
          the first pass's own S/N distribution (the `PIVSettings.sig2noise_threshold`
          field we'll set in notebook 3 comes from exactly this calculation).

        Next: **03_multipass_piv.py** — putting this into the full 4-pass
        windowed-deformation pipeline with `openpiv.windef`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
