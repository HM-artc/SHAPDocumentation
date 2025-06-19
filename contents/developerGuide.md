<frontmatter>
layout: default.md
title: "User Guide"
pageNav: 3
</frontmatter>

<br>

# Architecture Diagram

![SHAP Architecture Diagram](../src/shap_architecture.png)

# Class Diagram

<mermaid>
classDiagram

class NaiveSHAP {
+forecaster: MLForecast|StatsForecast|NeuralForecast
+futr_exog_list: list
+predict_fn(data: pd.DataFrame) pd.DataFrame
+expected_value: pd.DataFrame
+original_output: pd.DataFrame
+process_explanation() pd.DataFrame
+explain_single(model: str, unique_id: str, ds: pd.Timestamp) pd.Series
+plot_beeswarm()
+plot_waterfall(model: str, unique_id: str, ds: pd.Timestamp, max_display int)
-\_shap_cache: dict
-\_test_data: pd.DataFrame
-\_bg_means_num: pd.DataFrame
-\_model_group: ModelGroup
-\_hash_dataframe(df: pd.DataFrame): str
-shap_values(): pd.DataFrame
}

class ModelGroup {
+predict_fn(data: pd.DataFrame): pd.DataFrame
+expected_value: pd.DataFrame
+original_output: pd.DataFrame
+sample_data_average: pd.DataFrame
}

class MLForecastGroup { +(forecaster: MLForecast, train_data: pd.DataFrame, test_data: pd.DataFrame, feature_names: list)
+predict_fn(data: pd.DataFrame): pd.DataFrame
+expected_value: pd.DataFrame
+original_output: pd.DataFrame
+sample_data_average: pd.DataFrame
}

class NixtlaForecastGroup { +(forecaster: NeuralForecast|StatsForecast, test_data: pd.DataFrame, train_data: pd.DataFrame, futr_exog_list: list, context_data: pd.DataFrame=None)
+predict_fn(data: pd.DataFrame): pd.DataFrame
+expected_value: pd.DataFrame
+original_output: pd.DataFrame
+sample_data_average: pd.DataFrame
}

class ModelGroupFactory {
+create(forecaster, test_data: pd.DataFrame, train_data: pd.DataFrame, feature_names: list, context_data: pd.DataFrame=None) ModelGroup
}

class Plotter {
+beeswarm(shap_long: pd.DataFrame, order: list)
+waterfall(contrib: pd.Series, baseline: float, actual: float, title: str, max_display: int)
}

NaiveSHAP --> ModelGroupFactory
NaiveSHAP --> Plotter
MLForecastGroup --|> ModelGroup
NixtlaForecastGroup --|> ModelGroup
ModelGroupFactory ..> ModelGroup
</mermaid>

# Sequence Diagram

<mermaid>
sequenceDiagram
    participant User
    participant NaiveSHAP
    participant ModelGroupFactory
    participant ModelGroup
    participant Plotter

    User ->> NaiveSHAP: __init__(forecaster, test_data, train_data, futr_exog_list, context_data)
    NaiveSHAP ->> ModelGroupFactory: create(...)
    ModelGroupFactory -->> NaiveSHAP: ModelGroup instance

    User ->> NaiveSHAP: process_explanation()
    NaiveSHAP ->> NaiveSHAP: _hash_dataframe(test_data)
    alt cache miss
        NaiveSHAP ->> NaiveSHAP: shap_values()
        loop for each subset
            NaiveSHAP ->> ModelGroup: predict_fn(data)
        end
    end
    NaiveSHAP -->> User: combined SHAP DataFrame

    User ->> NaiveSHAP: explain_single(model, unique_id, ds)
    NaiveSHAP -->> User: pd.Series of φ_j

    User ->> NaiveSHAP: plot_beeswarm()
    NaiveSHAP ->> Plotter: beeswarm(shap_long, order)

    User ->> NaiveSHAP: plot_waterfall(model, unique_id, ds, max_display)
    NaiveSHAP ->> Plotter: waterfall(contrib, baseline, actual, title, max_display)

</mermaid>
