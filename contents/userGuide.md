<frontmatter>
layout: default.md
title: "User Guide"
pageNav: 3
</frontmatter>

# NaiveSHAP Class Documentation

## Overview

The `NaiveSHAP` class provides a model-agnostic, brute-force approximation of SHAP (SHapley Additive exPlanations) values for any of the supported forecaster types—**MLForecast**, **StatsForecast**, or **NeuralForecast**—over a fixed test horizon.  
It computes local feature attributions by perturbing one feature at a time (over all subsets), caches results for efficiency, and offers both **beeswarm** and **waterfall** visualizations via the `Plotter` module.

---

## Constructor

```python
NaiveSHAP(
    forecaster: MLForecast | StatsForecast | NeuralForecast,
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    futr_exog_list: list[str],
    context_data: pd.DataFrame = None
)
```

**Parameters:**

- **`forecaster`**  
  A trained forecasting object—one of:

  - `MLForecast`
  - `StatsForecast`
  - `NeuralForecast`

- **`test_data`** (`pd.DataFrame`)  
  The dataset (with columns `unique_id`, `ds`, and any exogenous features) whose forecasts you wish to explain.

- **`train_data`** (`pd.DataFrame`)  
  Historical data used to compute per-series averages for feature-value perturbations.

- **`futr_exog_list`** (`list[str]`)  
  Names of the exogenous feature columns in `test_data` to include in SHAP calculations.

- **`context_data`** (`pd.DataFrame`, optional)  
  For `NeuralForecast` models only: the rolling historical context required by RNN/TCN-style predictors.

---

## Public Methods

#### `process_explanation() → pd.DataFrame`

Compute (or fetch from cache) the full SHAP DataFrame. The returned DataFrame has columns:

- `unique_id`, `ds`
- `model`
- one column per feature in `futr_exog_list` containing the SHAP value φ
- `model_output` (the actual forecast)
- `expected_value` (the baseline forecast)

```python
shap_df = explainer.process_explanation()
```

---

#### `explain_single(model: str, unique_id: str, ds: pd.Timestamp) → pd.Series`

Fetch the SHAP values for one series-timestamp pair and a specific sub-model:

- **`model`**: name of the sub-model within `MLForecast` or the single output channel name for `StatsForecast`/`NeuralForecast`
- **`unique_id`**: series identifier
- **`ds`**: timestamp (must match the index in `test_data`)

Returns a `pd.Series` whose index is the feature list and whose values are the φ contributions.

```python
phi_series = explainer.explain_single(
    model="model_name",
    unique_id="series_1",
    ds=pd.Timestamp("2025-06-19")
)
```

Raises `ValueError` if no cached SHAP values exist or if the specified keys aren’t found.

---

#### `plot_beeswarm()`

Draw a SHAP **beeswarm** plot of all instances in the test set:

```python
explainer.plot_beeswarm()
```

- Requires prior call to `process_explanation()`.
- Uses `Plotter.beeswarm(shap_long, order)` under the hood.

---

#### `plot_waterfall(model: str, unique_id: str, ds: pd.Timestamp, max_display: int = None)`

Draw a SHAP **waterfall** for one instance:

- **`model`**: sub-model name
- **`unique_id`**, **`ds`**: identify the row to explain
- **`max_display`** (optional): limit to the top K features by absolute φ

```python
explainer.plot_waterfall(
    model="model_name",
    unique_id="series_1",
    ds=pd.Timestamp("2025-06-19"),
    max_display=10
)
```

- Requires prior call to `process_explanation()`.
- Uses `Plotter.waterfall(contrib, baseline, actual, title, max_display)` under the hood.

---

## Notes

- This implementation is _naive_ (brute-force): it exhaustively enumerates all subsets of features for each single-feature contribution.
- Caches the full SHAP DataFrame keyed by a hash of `test_data`, so repeated calls are cheap.
- Supports multi-model forecasts (e.g.\ ensemble members) transparently.
- Baseline (`expected_value`) is the model’s forecast when all exogenous features are set to their per-series averages over `train_data`.
- Ensures the **additivity** property:  
  \[
  f(x) = \mathbb{E}[f(X)] + \sum_j \phi_j.
  \]
