# Findings: openpiv-python and pivpy vs PIVlab

Notes collected while building the `notebooks/` tutorial series — comparing a modern
OpenPIV + pivpy workflow against PIVlab on the same von Kármán vortex street sequence.
Framed as candidate upstream issues, not filed automatically (see repo `CLAUDE.md`/README for
why).

## Bugs found

### 1. `pivpy.io`'s OpenPIV text reader inverts the mask column (confirmed, reproducible)

`openpiv.tools.save()` — what `openpiv.windef.piv()` uses to write `field_A%04d.txt` — writes
a 6th column where **`mask == 1` means "invalid / masked out"** (e.g. inside the cylinder) and
`mask == 0` means valid. That's the convention used consistently by `openpiv.tools.save` and
`openpiv.tools.display_vector_field`.

`pivpy.io.OpenPIVReader.read` treats that column as if it were a *validity* flag instead
(`invalid = mask == 0`) — the opposite convention. On our data, only ~2.7% of grid points are
inside the cylinder mask (`mask == 1`); the reader zeros out the other ~97% (the entire valid
flow field) and keeps the near-zero cylinder points. **Result: `ds['u']` and `ds['v']` come
back all zeros for every frame, with no error or warning** — this is the kind of bug a novice
would not catch, since the read "succeeds" and produces a plausible-looking (empty) dataset.

Reproduction: run `openpiv.windef.piv()` on any masked sequence, then
`pivpy.io.read_directory(results_dir, pattern="field_A*", ext=".txt")` and check
`np.abs(ds['u']).max() == 0`.

Workaround used in `notebooks/04_vorticity_with_pivpy.py` and `05_compare_to_pivlab.py`: load
with `numpy.loadtxt` and pass `chc = 1 - mask` into `pivpy.io.build_dataset` directly, bypassing
`OpenPIVReader`.

Candidate fix upstream: flip the sense in `OpenPIVReader.read`, or — better — make it configurable
/ documented, since "mask value that means invalid" is not universal across PIV tools (see
finding 2).

### 2. `pivpy.io.load_pivlab` doesn't handle PIVlab session `.mat` files

`pivpy.io.load_pivlab(filepath, frame=None)` expects `u_<n>`/`v_<n>`-style datasets (an older,
flat PIVlab export convention) and raises `ValueError: No u_<n> datasets found` on a PIVlab
**session file** (the default "Save session" `.mat`, MATLAB v7.3/HDF5), which instead stores a
`resultslist` cell array (one row per frame, columns x/y/u/v/typevector/...) plus a `derived`
cell array for whatever scalar field was selected in the UI when the session was saved. Session
files are the most common thing a PIVlab user actually has on disk (it's the default "save"
action), so this is a real gap for anyone trying to load PIVlab output into Python.

Workaround used in `notebooks/05_compare_to_pivlab.py`: read the session file directly with
`h5py`, indexing `resultslist[frame, col]` (HDF5 object references) for x/y/u/v.

Candidate fix upstream (also references pivpy issue #43 "How to replace `io.load_txt`?"): add
a `PIVLabSessionReader` alongside the existing `PIVLabReader`, detecting the file variant by its
top-level keys (`resultslist` present vs. `u_0` present).

## Performance: `validation.local_median_val` is the multipass pipeline's real bottleneck

Profiled a single image pair through the tutorial's 4-pass pipeline (64→32→16→6 px,
`correlation_method="linear"`, `normalized_correlation=True`, `smoothn=True`) with `cProfile`.
Total: ~52s instrumented (~35-40s real wall-clock). The biggest cost by far — **~33% of total
time** — is `openpiv.validation.local_median_val`, called once per pass (on `u` and `v`
separately, so 8 calls across 4 passes):

```python
um = generic_filter(masked_u, np.nanmedian, mode='constant', cval=np.nan, size=(3, 3))
```

`scipy.ndimage.generic_filter` with an arbitrary Python callable invokes that callable **once
per output pixel** — for our 169×340 finest-pass grid, that's 57,460 individual Python-level
calls into `np.nanmedian`. This is a well-known scipy anti-pattern: `generic_filter` exists for
filters with no vectorized equivalent, but a fixed-size local median has one. Direct
benchmark on a realistic grid:

| approach | time (169×340 grid, 3×3 window, one call) |
|---|---|
| `generic_filter(u, np.nanmedian, size=(3,3))` (current) | 3.2s |
| `sliding_window_view(u_padded, (3,3))` + `np.nanmedian(axis=(-2,-1))` | 0.067s |

Same result (`np.allclose(..., equal_nan=True)` → `True`), **~48x faster**, using only
`numpy.lib.stride_tricks.sliding_window_view` (no new dependency). Since this runs every pass
on every image pair, the effect compounds across a sequence — this single change would likely
cut multi-minute multipass runs dramatically more than any of the other candidates below.

Secondary costs, roughly in order (all much smaller than the above): `preprocess.dynamic_masking`'s
`scipy.ndimage.median_filter` (`filter_size=21`, ~9s for 2 calls, but fixed per-pair not
per-pass); `filters.replace_outliers` → `lib.replace_nans` (iterative hole-filling, also
per-pass). Note `use_vectorized=True` (toggles `vectorized_correlation_to_displacements` for
the subpixel fit) only gave ~8% wall-clock improvement in a direct A/B timing (43.8s → 40.4s)
despite looking larger in the profiler — profiler self-time overinflates Python-loop-heavy code,
so don't trust `cProfile` percentages alone for anything callback-heavy; time it directly too.

Reproduction: `cProfile.run(...)` around `windef.piv(settings)` on any multi-pass sequence;
`local_median_val` / `generic_filter` dominates `cumulative` time. Candidate fix upstream:
replace the `generic_filter(..., np.nanmedian, ...)` call in `validation.local_median_val` with
a `sliding_window_view`-based vectorized local median.

## Coordinate-convention gotcha (not a bug, but a trap for novices)

PIVlab keeps velocity/vorticity in image-pixel convention (`y` increasing downward, `v` in
image-row direction). `openpiv.windef.piv()` applies `transform_coordinates()`, which flips the
row order of `y` and negates `v` to produce a right-handed "physical" (`y`-up) frame. Comparing
the two tools' output without replicating that transform on the PIVlab side gives a **near-zero
correlation** even though both tools tracked the same flow correctly — it looks like the tools
disagree when the real issue is an unmatched axis convention. `05_compare_to_pivlab.py`
demonstrates the fix (match the transform) and, as a general practice, checks agreement on the
raw `u, v` fields before trusting any derived-quantity (vorticity) comparison — see next point.

## UX / novice-friendliness observations

- **Differentiated quantities (vorticity) are a bad first comparison metric.** Two
  independently-computed, independently-masked velocity fields can agree well in `u, v` and
  still show a weak pointwise vorticity correlation, because differentiation amplifies small
  grid/masking misalignments. A tutorial (or a user) that compares tools on vorticity first
  will likely conclude "these disagree" when the underlying velocity fields actually agree
  fine. Worth a callout in OpenPIV/pivpy docs: validate on `u, v` before judging a derived
  field.
- **openpiv-python's API churn between versions has no migration notes for existing
  tutorials.** `windef.Settings` → `windef.PIVSettings`, `iterations` → `num_iterations`, and
  the old manual `first_pass` + `multipass_img_deform` loop collapsed into one `windef.piv()`
  call. All good simplifications, but the pre-existing `openpiv-python-examples` notebooks
  (including ones nearly identical to this repo's former `OpenPIV/` notebooks) were not
  updated in lockstep, so a newcomer following an old tutorial hits `AttributeError:
  'module' has no attribute 'Settings'` with no pointer to what changed. Relates to
  openpiv-python issue #293 (config-driven `PIVSettings` instead of hardcoded — would also make
  version migrations less painful, since a schema/version field could flag renamed keys).
- **`PIVSettings` has no validation** — e.g. nothing stops `windowsizes` and `overlap` tuples
  from having mismatched lengths, or `overlap >= windowsizes[i]`, both of which fail deep inside
  the multipass loop with a stack trace that doesn't point at the setting. A `__post_init__`
  sanity check (or at minimum a clearer error at the point of use) would help newcomers a lot —
  relates to openpiv-python issue #293.
- **`smoothn.py`** (still used by `windef.piv()` when `settings.smoothn=True`) is a fairly
  opaque hand-ported MATLAB algorithm; openpiv-python issue #268 ("replace smoothn.py with
  scipy.interpolate") would make this step easier to reason about/debug for contributors.
- **No marimo examples existed in `openpiv-python-examples`** prior to this tutorial; pivpy
  already has one (`examples/notebooks/Getting_Started.py`, with an "Open in molab" badge).
  Worth proposing this tutorial (or a trimmed version) as a contribution to
  `openpiv-python-examples`, matching pivpy's existing marimo/molab precedent.
- **No output format unification** between OpenPIV's plain-text columns and pivpy's canonical
  xarray schema meant we needed the loader workarounds above — this is exactly openpiv-python
  issue #204 ("Unifying output of OpenPIV").

## Forward-looking note

pivpy is under active development; a nicer `ds.piv.plot()` (better default colormap) is planned/
in progress upstream but not yet in the PyPI release this tutorial pins (`0.1.2` — `plot` is
present only as a commented-out stub in `pivpy/pivpy.py` there). The notebooks here use
`ds.piv.vorticity()` + manual `matplotlib.pcolormesh` (for the fixed ±0.3 scale needed to match
the original README) and `showscal`/`quiver` for everything else. Once `.piv.plot()` ships in a
release, `04_vorticity_with_pivpy.py`'s plotting cell is the one to revisit.

## Seed issues referenced above

- openpiv-python #293 — YAML/JSON config instead of hardcoded `PIVSettings`
- openpiv-python #204 — unify OpenPIV's output formats
- openpiv-python #268 — replace `smoothn.py` with `scipy.interpolate`
- openpiv-python #152 — old untriaged bug report (not revisited in depth here; flagged for
  follow-up)
- pivpy #43 — "How to replace `io.load_txt`?" (directly relevant to finding 2 above)
- pivpy #28 — auto `.fillna(0)` in plots (beginner-friendliness; not hit directly in this
  tutorial, but consistent with the "silent wrong-looking output" theme of finding 1)
