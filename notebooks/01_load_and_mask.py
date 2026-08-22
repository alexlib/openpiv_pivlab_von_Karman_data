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
    from openpiv import tools, preprocess

    DATA_DIR = Path(__file__).parent.parent / "data"
    return DATA_DIR, mo, np, plt, preprocess, tools


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Loading images and masking the cylinder

    This tutorial series walks through a modern OpenPIV + pivpy analysis of a
    von Kármán vortex street, and compares it against PIVlab (MATLAB) results
    on the same image sequence.

    The images show flow past a fixed cylinder. Before we can run PIV, we need
    to **mask the cylinder** — otherwise OpenPIV will try (and fail) to find a
    correlation peak inside a solid, non-moving object, contaminating the
    results near the body.

    OpenPIV can derive this mask automatically from image intensity via
    `openpiv.preprocess.dynamic_masking`. The catch: the images also contain
    reflections/shadows around the cylinder that look similar in intensity —
    a naive threshold masks those too, throwing away valid vectors. Getting
    the mask right (cylinder only, not the shadow) is the first parameter
    choice you have to make, and it's a good first taste of "PIV is not
    push-button."
    """)
    return


@app.cell
def _(DATA_DIR):
    pair_ids = sorted(
        p.name[len("karman_16Hz_"):-len("_A.jpg")]
        for p in DATA_DIR.glob("karman_16Hz_*_A.jpg")
    )
    return (pair_ids,)


@app.cell
def _(mo, pair_ids):
    pair_picker = mo.ui.dropdown(options=pair_ids, value=pair_ids[0], label="image pair")
    pair_picker
    return (pair_picker,)


@app.cell
def _(DATA_DIR, pair_picker, tools):
    frame_a = tools.imread(DATA_DIR / f"karman_16Hz_{pair_picker.value}_A.jpg")
    frame_b = tools.imread(DATA_DIR / f"karman_16Hz_{pair_picker.value}_B.jpg")
    return frame_a, frame_b


@app.cell
def _(frame_a, plt):
    fig_raw, ax_raw = plt.subplots(figsize=(8, 4))
    ax_raw.imshow(frame_a, cmap="gray")
    ax_raw.set_title("Frame A (raw) — find the cylinder and its shadow")
    fig_raw
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tune the masking parameters

    `dynamic_masking` has three knobs:

    - **method** — `"intensity"` thresholds pixel brightness directly;
      `"edges"` masks based on edge detection. `"intensity"` works well here
      because the cylinder is much darker/brighter than the background.
    - **threshold** — fraction of the intensity range that counts as
      "object". Too low and you mask nothing; too high and you eat into the
      shadow (and lose valid vectors there).
    - **filter_size** — size of the smoothing filter applied before
      thresholding; larger values give a smoother, less noisy mask outline
      but can blur the true edge of the cylinder.

    Move the sliders and watch the mask overlay below. **Goal: the red mask
    should hug the cylinder outline and stop there — not bleed into the
    surrounding flow.**
    """)
    return


@app.cell
def _(mo):
    method_picker = mo.ui.dropdown(
        options=["intensity", "edges"], value="intensity", label="masking method"
    )
    threshold_slider = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.1, label="threshold"
    )
    filter_size_slider = mo.ui.slider(
        start=3, stop=41, step=2, value=21, label="filter_size"
    )
    mo.hstack([method_picker, threshold_slider, filter_size_slider])
    return filter_size_slider, method_picker, threshold_slider


@app.cell
def _(
    filter_size_slider,
    frame_a,
    frame_b,
    method_picker,
    np,
    preprocess,
    threshold_slider,
):
    masked_a, mask_a = preprocess.dynamic_masking(
        frame_a,
        method=method_picker.value,
        filter_size=filter_size_slider.value,
        threshold=threshold_slider.value,
    )
    masked_b, mask_b = preprocess.dynamic_masking(
        frame_b,
        method=method_picker.value,
        filter_size=filter_size_slider.value,
        threshold=threshold_slider.value,
    )
    # a pixel is masked only where BOTH frames agree it's the (stationary) body
    image_mask = np.logical_and(mask_a, mask_b)
    return (image_mask,)


@app.cell
def _(frame_a, image_mask, np, plt):
    fig_mask, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 4))
    ax_before.imshow(frame_a, cmap="gray")
    ax_before.set_title("before masking")

    overlay = np.dstack([frame_a, frame_a, frame_a]).astype(float)
    overlay[..., 0] = np.clip(overlay[..., 0] + 120 * image_mask, 0, 255)
    ax_after.imshow(overlay.astype("uint8"))
    ax_after.set_title(f"mask overlay ({image_mask.sum()} px masked)")
    fig_mask
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaway

    `image_mask` (the logical AND of both frames' masks) is what feeds into
    the PIV settings in the next notebook, as
    `settings.dynamic_masking_method` / `_threshold` / `_filter_size`. There
    is no formula for the "right" threshold — you tune it visually, the same
    way you would in PIVlab's interactive mask editor. The difference is that
    here the tuning loop is a slider, not a modal dialog, and the mask is
    automatically recomputed and reused for every image pair in the sequence
    rather than drawn once by hand.

    Next: **02_choosing_parameters.py** — picking window sizes, overlap, and
    the signal-to-noise threshold.
    """)
    return


if __name__ == "__main__":
    app.run()
