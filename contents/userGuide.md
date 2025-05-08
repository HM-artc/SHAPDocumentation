<frontmatter>
layout: default.md
title: "User Guide"
pageNav: 3
</frontmatter>

# NaiveSHAP Class Documentation

## Overview

The `NaiveSHAP` class provides a simple, efficient approach to approximating SHAP (SHapley Additive exPlanations) values for any MLForecast or NeuralForecast forecaster. It supports local explanations, feature attribution, and visualizations.

---

## Constructor

`NaiveSHAP(forecaster, background_data, futr_exog_list, futr_exogenous_data=None, model_key=None, model_idx=None)`

**Parameters:**

- `forecaster`: The trained forecasting model (`MLForecast` or `NeuralForecast` object).
- `background_data` (pd.DataFrame): Background dataset to explain over.
- `futr_exog_list` (list): List of future exogenous feature names to perturb.
- `model_key` (str, optional): Name of the model inside an MLForecast instance.
- `model_idx` (int, optional): Index of the model inside a NeuralForecast instance.

**Description:**  
Initializes internal attributes, sets up model-specific prediction handlers, and prepares for SHAP computation.

---

## Methods

`process_explanation() -> pd.DataFrame`

Computes SHAP values (if not cached) and returns a DataFrame that includes:

- SHAP values
- Expected value (baseline prediction)
- Model outputs for each row

`explain_single(index: int) -> pd.Series`

Explains a single instance by returning a sorted list of feature contributions.

- **index**: Integer index of the instance to explain.

Raises `ValueError` if no SHAP values computed yet.  
Raises `IndexError` if index is out of range.

`explain_single_detailed(index: int) -> str`

Explains a single instance by returning a sorted list of feature contributions, the expected value by the model and the actual prediction value of chosen point.

- **index**: Integer index of the instance to explain.

Raises `ValueError` if no SHAP values computed yet.  
Raises `IndexError` if index is out of range.

`plot_beeswarm()`

Plots a **beeswarm plot** of feature impacts across all instances.

- Requires prior computation of SHAP values.
- Uses the `Plotter.beeswarm` method.

`plot_waterfall(index: int, max_display: int = None)`

Plots a **waterfall plot** for a given instance:

- **index**: Integer index of the instance.
- **max_display** (optional): Maximum number of features to display in the plot.
- Requires prior computation of SHAP values.
- Uses the `Plotter.waterfall` method.

---

## Notes

- The method is _naive_ because it uses leave-one-out perturbations without considering feature interactions and uses a brute force method to calculate the SHAP values.
- Normalization ensures SHAP additivity property is maintained.
- Works for both MLForecast and NeuralForecast models.
