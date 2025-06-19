<frontmatter>
  title: Home Page
  layout: default.md
  pageNav: 4
  pageNavTitle: "Topics"
</frontmatter>

<br>

# Project Introduction

Black box models have achieved remarkable predictive performance but are difficult for users and stakeholders to trust, audit, or understand their decisions.

SHAP (SHapley Additive exPlanations) provides a unified, game-theoretic approach to interpreting the output of machine learning models.

While there exists SHAP libraries, internal literature review has shown it to be overly complex for development with Nixtla time series forecasting models.

## Project Objective

This project implements a lightweight and efficient **Naive SHAP framework** tailored for:

- **MLForecast** models
- **NeuralForecast** models

It enables users to:

- Compute approximate SHAP values using a simple leave-one-out methodology
- Visualize feature attributions with **beeswarm** and **waterfall** plots
- Support both tabular and time-series explainability out-of-the-box

## Project Results

A demo run can be found in `main.py`.

The demo run uses a synthetic time series dataset consisting of two exogenous features: `lag1` and `dayofweek`. A `Neuralforecast` and `Statsforecast` models have been trained
using this data. A forecast of 12 weeks have been made using each of these models.

The `NaiveShap` class have been applied to both these models for explanability of the 12 week forecast.

A summary of all SHAP results, a visualisation of this in a beeswarm plot, a point SHAP value and a visualisation of this in a waterfall plot.

#### Summary Table

Summary table for Statsforecast:

| dayofweek | lag1      | model     | unique_id | ds         | model_output | expected_value |
| --------- | --------- | --------- | --------- | ---------- | ------------ | -------------- |
| -2.96507  | 4.041896  | AutoARIMA | id_0      | 2000-07-29 | 0.63585      | -0.440989      |
| -3.95341  | -3.8764   | AutoARIMA | id_0      | 2000-07-30 | -0.150702    | 7.679108       |
| 1.976075  | -2.360005 | AutoARIMA | id_0      | 2000-07-31 | 4.204175     | 4.587475       |
| 0.988352  | -1.368008 | AutoARIMA | id_0      | 2000-08-01 | 2.603441     | 2.983097       |
| 0.0       | -0.278635 | AutoARIMA | id_0      | 2000-08-02 | 2.889822     | 3.16846        |
| -0.988352 | 1.401491  | AutoARIMA | id_0      | 2000-08-03 | 5.269206     | 4.839067       |
| -1.976705 | 2.386555  | AutoARIMA | id_0      | 2000-08-04 | 7.345608     | 6.93517        |
| -2.965057 | 3.666592  | AutoARIMA | id_0      | 2000-08-05 | 1.624613     | 0.929378       |
| -3.95341  | -3.646156 | AutoARIMA | id_0      | 2000-08-06 | -3.016255    | 4.58331        |
| 1.976075  | -2.255879 | AutoARIMA | id_0      | 2000-08-07 | 5.021028     | 5.300203       |
| 0.988352  | -1.143153 | AutoARIMA | id_0      | 2000-08-08 | 4.289725     | 4.427088       |
| 0.0       | 0.249522  | AutoARIMA | id_0      | 2000-08-09 | 3.432135     | 3.126213       |
| 3.012113  | 0.020223  | AutoARIMA | id_1      | 2000-03-26 | 6.278505     | 3.24627        |
| 2.008075  | 0.03243   | AutoARIMA | id_1      | 2000-03-27 | 0.266857     | 3.24627        |
| -2.008075 | -0.025651 | AutoARIMA | id_1      | 2000-03-28 | 2.125244     | 3.24627        |
| -1.00048  | -0.031647 | AutoARIMA | id_1      | 2000-03-29 | 2.229186     | 3.24627        |
| 0.0       | -0.006759 | AutoARIMA | id_1      | 2000-03-30 | 3.239511     | 3.24627        |
| 1.00048   | 0.002609  | AutoARIMA | id_1      | 2000-03-31 | 4.252377     | 3.24627        |
| 2.008075  | 0.011252  | AutoARIMA | id_1      | 2000-04-01 | 5.265597     | 3.24627        |
| 3.012113  | 0.020493  | AutoARIMA | id_1      | 2000-04-02 | 6.281431     | 3.24627        |
| -3.012113 | 0.030236  | AutoARIMA | id_1      | 2000-04-03 | 0.264393     | 3.24627        |
| -2.008075 | -0.023782 | AutoARIMA | id_1      | 2000-04-04 | 1.214413     | 3.24627        |
| -1.00048  | -0.014021 | AutoARIMA | id_1      | 2000-04-05 | 2.228112     | 3.24627        |
| 0.0       | -0.003499 | AutoARIMA | id_1      | 2000-04-06 | 3.242771     | 3.24627        |

#### Beeswarm plot

Beeswarm plot for Statsforecast
![Beeswarm plot for Statsforecast](./src/sf_beeswarm.png)

#### Point SHAP value

Point table for Statsforecast:

| dayofweek |     lag1 | model     | unique_id | ds                  | model_output | expected_value |
| --------: | -------: | :-------- | :-------- | :------------------ | -----------: | -------------: |
|         0 | 0.249522 | AutoARIMA | id_0      | 2000-08-09 00:00:00 |      3.43214 |        3.12621 |

#### Waterfall Plot

Waterfall plot for Statsforecast
![Waterfall plot for Statsforecast](./src/sf_waterfall.png)

## Benchmark Testing

The internal SHAP implementation has been benchmarked against the [standard SHAP library](https://shap.readthedocs.io/en/latest/). The preparation for the test follows the procedure outlined in the [Nixtla documentation](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/analyzing_models.html).

Using a paired t-test, and assuming an alternate hypothesis that the SHAP values output by the internal implementation is different from the standard SHAP implementation, we find a p-value as listed in this table:

|               model | p-value |
| ------------------: | ------: |
|   Linear Regression |  0.8225 |
|               Ridge |  0.8222 |
|               Lasso |  0.9297 |
| KNeighborsRegressor |  0.3758 |
|          ElasticNet |  0.6986 |

We do not reject the null hypothesis at 1% significance level.

The test may be run from the root of the folder using the following command:

```
python -m unittest
```

## Future Work

- Integrate faster approximation methods (e.g., KernelSHAP-like sampling)
- SHAP for multi-output models
