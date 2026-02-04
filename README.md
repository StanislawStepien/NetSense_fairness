# NetSense Fairness Pipeline

## Access Note
In `query.py`, the function `run_query_return_df` requires a credentials JSON file: `netsense-411221-be346442c94a.json`. For security, it is not included in this repo. If you need it, please contact stanislaw.stepien@pwr.edu.pl.

## Feature Engineering (Survey + Network)
Feature engineering produces two core tables used in the hybrid pipeline:

- `survey_and_hybrid_approach_pipeline/pure_demographical_data.csv`
- `survey_and_hybrid_approach_pipeline/all_features_plus_neighbours.csv`

### What It Does (and Why)
1. **Pure demographics (survey-based)**
   - Extracts stable demographic variables across survey waves.
   - Reshapes multi-wave columns into per-survey rows (`SurveyNr`).
   - Bins numeric variables into interpretable ranges (weight, height, computer use, number of pets).
   - Encodes categorical variables and imputes missing values to reduce sparsity.

2. **Topology features (network-based)**
   - Reshapes network survey data into per-survey rows.
   - Builds per-survey ego networks from `EgoID`–`AlterID` interactions.
   - Computes centrality metrics per ego (degree, closeness, betweenness, eigenvector, community).

3. **Merge**
   - Merges engineered survey features with topology features on `egoid` + `SurveyNr`.

### Run Feature Engineering
- `python feature_engineering/run_feature_engineering.py --config feature_engineering/feature_engineering_config.json`

Optional flags:
- `--survey-features` to point to a precomputed survey feature table
- `--skip-topology` to reuse existing `netsense/topology_based_features.csv`
- `--analysis` to write a feature-engineering report (summary stats, correlations, outliers)
- `--plots` to save plots (histograms, correlation heatmaps, boxplots)

## Survey + Hybrid Pipeline (Training + Evaluation)
The reproducible training pipeline lives in `survey_and_hybrid_approach_pipeline/` and is container-ready.

### Inputs
- `survey_and_hybrid_approach_pipeline/all_features_plus_neighbours.csv`
- `survey_and_hybrid_approach_pipeline/pure_demographical_data.csv`
- `netsense/topology_based_features.csv` (hybrid only)
- `simulation_result_per_topic/best_sim_result/*.csv`
- `survey_and_hybrid_approach_pipeline/dictionary_of_dfs_with_minorities.pkl`

### Outputs
- Models: `survey_and_hybrid_approach_pipeline/models/*_best_model_for_selected_features.joblib`
- Evaluation results: `survey_and_hybrid_approach_pipeline/evaluation_results/evaluation_results_<topic>.csv`
- Artifacts: `survey_and_hybrid_approach_pipeline/artifacts/best_for_topic.joblib` and `.json`

### Pipeline Steps
1. Load demographic features and pure demographic features.
2. For hybrid runs, merge topology features from `netsense/topology_based_features.csv`.
3. Drop opinion target columns from the feature matrix.
4. Load simulation results per topic and normalize identifiers.
5. Merge simulation outcomes with features and create label `Y`.
6. Build balanced train/test splits for minority and non‑minority groups.
7. Model comparison step (experiments):
   - Survey-only: RandomForest, balanced RandomForest, DecisionTree.
   - Hybrid: balanced RandomForest, DecisionTree.
8. Select the best-performing model based on evaluation metrics.
9. Run Bayesian hyperparameter search for the selected model (if RandomForest).
10. Evaluate top‑K feature subsets and choose best.
11. Evaluate minority performance by group and save results.
12. Save best models and artifacts per topic.

### Current Best Model (From Experiments)
- Survey-based features (without topology): `RandomForestClassifier` with `class_weight="balanced"`.

### Run Training
- `python survey_and_hybrid_approach_pipeline/run_pipeline.py --config survey_and_hybrid_approach_pipeline/config.json`

### Run Survey + Hybrid Pipeline (End-to-End)
- `python run_survey_and_hybrid_pipeline.py --fe-config feature_engineering/feature_engineering_config.json --train-config survey_and_hybrid_approach_pipeline/config.json`

Examples:
- Without analysis: `python run_survey_and_hybrid_pipeline.py --fe-config feature_engineering/feature_engineering_config.json --train-config survey_and_hybrid_approach_pipeline/config.json`
- With analysis: `python run_survey_and_hybrid_pipeline.py --fe-config feature_engineering/feature_engineering_config.json --train-config survey_and_hybrid_approach_pipeline/config.json --analysis`
- With analysis + plots: `python run_survey_and_hybrid_pipeline.py --fe-config feature_engineering/feature_engineering_config.json --train-config survey_and_hybrid_approach_pipeline/config.json --analysis --plots`

### Container Run
- `docker build -t netsense-pipeline .`
- `docker run --rm -v "$(pwd)":/app netsense-pipeline`
