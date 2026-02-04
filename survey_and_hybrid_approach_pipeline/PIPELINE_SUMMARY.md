# Survey + Hybrid Pipeline Summary

## Purpose
This pipeline trains and evaluates per-topic classifiers that predict mismatches between simulation and survey opinions. It combines demographic features (and topology-based features for the hybrid run) with simulation outputs, then evaluates model performance for minority subgroups.

## Feature Engineering Inputs
- `netsense/demsurveyMergedCodedDisID.csv`
- `netsense/netsurveysMergedWideCodedWithDates.csv`
- `netsense/topology_based_features.csv`

## Training Inputs
- `survey_and_hybrid_approach_pipeline/all_features_plus_neighbours.csv`
- `survey_and_hybrid_approach_pipeline/pure_demographical_data.csv`
- `netsense/topology_based_features.csv`
- `simulation_result_per_topic/best_sim_result/*.csv`
- `survey_and_hybrid_approach_pipeline/dictionary_of_dfs_with_minorities.pkl`

## Outputs
- Models: `survey_and_hybrid_approach_pipeline/models/*_best_model_for_selected_features.joblib`
- Evaluation results: `survey_and_hybrid_approach_pipeline/evaluation_results/evaluation_results_<topic>.csv`
- Artifacts: `survey_and_hybrid_approach_pipeline/artifacts/best_for_topic.joblib` and `.json`

## Feature Engineering Steps (What + Why)
- Reshape multi-wave survey data into per-survey rows (`SurveyNr`).
- Extract stable demographic variables and bin numeric values into interpretable ranges.
- Encode categorical variables and fill missing values to reduce sparsity.
- Build per-survey ego networks and compute centrality metrics.
- Merge engineered survey features with topology features by `egoid` + `SurveyNr`.

## Training Pipeline Steps
- Load demographic features and pure demographic features.
- For hybrid runs, merge topology features from `netsense/topology_based_features.csv`.
- Drop opinion target columns from the feature matrix.
- Load simulation results per topic and normalize identifiers.
- Merge simulation outcomes with features and create label `Y`.
- Build balanced train/test splits for minority and non-minority groups.
- Model comparison step (experiments).
- Select the best-performing model based on evaluation metrics.
- Run Bayesian hyperparameter search for the selected model (current default: balanced RandomForest).
- Evaluate top-K feature subsets and choose best.
- Evaluate minority performance by group and save results.
- Save best models and artifacts per topic.

## Model Comparison (Experiments)
- Survey-only features (no topology)
  - `RandomForestClassifier`
  - `RandomForestClassifier` with `class_weight="balanced"` (stratified/balanced)
  - `DecisionTreeClassifier`
- Hybrid features (demographic + topology)
  - `RandomForestClassifier` with `class_weight="balanced"`
  - `DecisionTreeClassifier`

## Current Best Model (From Experiments)
- Survey-based features (without topology): `RandomForestClassifier` with `class_weight="balanced"`.

## How To Run
1. Feature engineering:
   - `python feature_engineering/run_feature_engineering.py --config feature_engineering/feature_engineering_config.json`
2. Training:
   - `python survey_and_hybrid_approach_pipeline/run_pipeline.py --config survey_and_hybrid_approach_pipeline/config.json`
3. End-to-end (feature engineering + training):
   - `python run_survey_and_hybrid_pipeline.py --fe-config feature_engineering/feature_engineering_config.json --train-config survey_and_hybrid_approach_pipeline/config.json`

## Known Gaps From The Original Notebook
- The notebook referenced a topology CSV in a different folder. This pipeline uses `netsense/topology_based_features.csv`.
- The notebook referenced simulation results under a different path; this repo uses `simulation_result_per_topic/best_sim_result`.
- The notebook wrote models/results outside the repo. This pipeline keeps outputs inside `survey_and_hybrid_approach_pipeline/`.
- The notebook’s late evaluation cells referenced undefined variables; they are replaced by consistent evaluation here.
