# Plasmon XRTS Observable Layers

This note documents the observable reconstruction layers that sit above the
existing plasmon dielectric backends.

## Purpose

The historical plasmon benchmark path compared a peak extracted directly from the
backend free-electron loss / DSF response. That is appropriate for backend-level
dielectric validation, but it is not the same level as an experiment-facing XRTS
observable.

The current code now exposes three explicit comparison levels so the same backend
response can be evaluated at the right layer:

- `dielectric`: backend free-electron spectrum only
- `xrts_observable`: minimal Chihara-like Al observable control
- `xrts_article_native_al`: material-specific / article-native Al observable experiment

The goal is benchmark honesty, not curve fitting.

## Architecture

Relevant files:

- [`src/helios/services/derived/plasmon.py`](../src/helios/services/derived/plasmon.py)
- [`src/helios/services/derived/plasmon_xrts_observable.py`](../src/helios/services/derived/plasmon_xrts_observable.py)
- [`src/helios/services/derived/plasmon_xrts_material.py`](../src/helios/services/derived/plasmon_xrts_material.py)
- [`scripts/benchmark_plasmon_article_cases.py`](../scripts/benchmark_plasmon_article_cases.py)
- [`scripts/benchmark_plasmon_xrts_observable.py`](../scripts/benchmark_plasmon_xrts_observable.py)
- [`scripts/benchmark_plasmon_article_native_observable.py`](../scripts/benchmark_plasmon_article_native_observable.py)

The backend still owns the dielectric response:

1. backend computes `chi(q, omega)` and `epsilon(q, omega)`
2. loss function and free-electron DSF are derived
3. the observable layer optionally reconstructs a material-specific XRTS signal
4. the final observable is convolved and normalized
5. peak extraction is performed on the final spectrum

This keeps backend physics separate from observable construction and allows the
same backend to be compared both at the dielectric level and at the
experiment-facing observable level.

## Current decomposition modes

### `xrts_observable`

The minimal control layer remains intentionally conservative and explicit:

- material-specific target: `Al`
- decomposition style: minimal Chihara-like
- free-electron term: backend DSF
- elastic / ion-feature term: compact bound-electron form-factor proxy centered at zero energy transfer
- bound/core inelastic term: explicit but currently zero inside the narrow article benchmark window below the first Al L-shell onset

### `xrts_article_native_al`

The article-native Al layer upgrades the observable assembly without touching the
backend dielectric:

- material-specific target: `Al`
- decomposition style: article-facing Al Chihara-like assembly
- free-electron term: backend DSF
- elastic / ion-feature term: explicit Al Cromer-Mann form-factor bookkeeping
- bound/core bookkeeping: neutral-Al versus Al3+ split plus explicit shell-threshold diagnostics
- bound/core inelastic term: still zero inside the current benchmark window because `|omega| < 72.6 eV` stays below the first Al L-shell onset
- comparison peak: extracted on the positive inelastic branch after explicit elastic subtraction

For unsupported or mixed-material subsets, the code falls back to the backend
free-electron spectrum with explicit provenance instead of inventing hidden atomic
physics.

## Peak extraction

In both observable modes, the final reported benchmark peak is no longer taken
from the raw total spectrum maximum.

- `xrts_observable`: positive inelastic branch after excluding a central elastic-core window
- `xrts_article_native_al`: positive inelastic branch after explicit elastic subtraction, with the exclusion window and fit/fallback state reported in provenance

This is a deliberate comparison-level change from the dielectric mode.

## Usage

Inside the Derived / Analysis plasmon UI:

- choose the backend model as usual
- choose `Observable = Dielectric-only`, `XRTS observable`, or `XRTS article-native Al`

From the benchmark CLI:

```powershell
python scripts\benchmark_plasmon_article_cases.py --dataset 50Al+10E+25CH+3.5TW_stabilized.h5 --observable-mode dielectric
python scripts\benchmark_plasmon_article_cases.py --dataset 50Al+10E+25CH+3.5TW_stabilized.h5 --observable-mode xrts_observable
python scripts\benchmark_plasmon_article_cases.py --dataset 50Al+10E+25CH+3.5TW_stabilized.h5 --observable-mode xrts_article_native_al
python scripts\benchmark_plasmon_xrts_observable.py --dataset 50Al+10E+25CH+3.5TW_stabilized.h5
python scripts\benchmark_plasmon_article_native_observable.py --dataset 50Al+10E+25CH+3.5TW_stabilized.h5
```

## Output artifacts

The minimal-control observable pass writes:

- `outputs/validation_outputs/plasmon_article_observable_audit.md`
- `outputs/validation_outputs/plasmon_xrts_observable_experiment/report.md`
- `outputs/validation_outputs/plasmon_xrts_observable_experiment/benchmark_summary.csv`
- `outputs/validation_outputs/plasmon_xrts_observable_experiment/response_model_comparison.csv`
- `outputs/validation_outputs/plasmon_xrts_observable_experiment/xrts_spectra_comparison.csv`
- `outputs/validation_outputs/plasmon_xrts_observable_experiment/component_breakdown.csv`
- `outputs/validation_outputs/plasmon_xrts_observable_experiment/control_vs_observable_delta.csv`
- `outputs/validation_outputs/plasmon_xrts_observable_experiment/observable_provenance.csv`
- `outputs/validation_outputs/plasmon_xrts_observable_experiment/convolution_sensitivity.csv`
- `outputs/validation_outputs/plasmon_xrts_next_step.md`

The article-native Al observable pass writes:

- `outputs/validation_outputs/plasmon_article_native_observable_audit.md`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/report.md`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/article_native_observable_summary.csv`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/control_vs_article_native_delta.csv`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/component_breakdown.csv`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/observable_provenance.csv`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/normalization_subtraction_diagnostics.csv`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/elastic_feature_diagnostics.csv`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/bound_core_diagnostics.csv`
- `outputs/validation_outputs/plasmon_article_native_observable_experiment/xrts_spectra_comparison.csv`
- `outputs/validation_outputs/plasmon_article_native_next_step.md`

## Diagnostics and limits

The observable layers report:

- selected backend
- observable mode
- decomposition mode
- peak-extraction mode
- elastic exclusion window
- free / bound / elastic fractions
- normalization / subtraction mode
- article-native Al elastic form-factor diagnostics
- shell-threshold bound/core bookkeeping

Current limits:

- the article-native Al layer still does not contain a real bound-free Al cross section
- the ion feature still uses the `S_ii(q)=1` assumption
- exact article-side background subtraction and normalization conventions are not recoverable from the current repo alone
- the layer improves honesty and interpretability of the observable comparison, but it does not replace a full article-native atomic / instrument forward model
