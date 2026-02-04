# Feature Engineering

This folder contains a runnable feature-engineering pipeline that prepares the two core feature tables used by the survey + hybrid pipeline:

- `survey_and_hybrid_approach_pipeline/pure_demographical_data.csv`
- `survey_and_hybrid_approach_pipeline/all_features_plus_neighbours.csv`

## What It Does (and Why)
1. **Pure demographics (survey-based)**
   - Extracts a stable subset of demographic variables (e.g., parental education, ethnicity, disability, income, etc.).
   - Reshapes multi-wave survey columns into per-survey rows (`SurveyNr`).
   - Bins numeric variables into interpretable ranges (e.g., weight, height, computer use).
   - Fills missing values with the most frequent category and encodes categorical variables.

2. **Topology features (network-based)**
   - Reshapes network survey data into per-survey rows.
   - Builds per-survey ego networks from `EgoID`–`AlterID` interactions.
   - Computes network metrics per ego (degree, closeness, betweenness, eigenvector centrality, community assignment).

3. **Merge**
   - Merges the engineered survey features with topology features on `egoid` + `SurveyNr`.

## How To Run
From the repo root:

- `python feature_engineering/run_feature_engineering.py --config feature_engineering/feature_engineering_config.json`

If you already have a precomputed survey features table (as in this repo), you can pass it explicitly:

- `python feature_engineering/run_feature_engineering.py --config feature_engineering/feature_engineering_config.json --survey-features survey_and_hybrid_approach_pipeline/all_features_plus_neighbours.csv`

To skip recomputing topology features (use existing `netsense/topology_based_features.csv`):

- `python feature_engineering/run_feature_engineering.py --config feature_engineering/feature_engineering_config.json --skip-topology`

To write an analysis report (summary stats, correlations, and outliers):

- `python feature_engineering/run_feature_engineering.py --config feature_engineering/feature_engineering_config.json --analysis`

To write plots (histograms, correlations, boxplots):

- `python feature_engineering/run_feature_engineering.py --config feature_engineering/feature_engineering_config.json --plots`

To write both analysis and plots:

- `python feature_engineering/run_feature_engineering.py --config feature_engineering/feature_engineering_config.json --analysis --plots`

## Inputs (Configured in `feature_engineering/feature_engineering_config.json`)
- `netsense/demsurveyMergedCodedDisID.csv`
- `netsense/netsurveysMergedWideCodedWithDates.csv`
- `netsense/topology_based_features.csv`

## Outputs
- `survey_and_hybrid_approach_pipeline/pure_demographical_data.csv`
- `survey_and_hybrid_approach_pipeline/all_features_plus_neighbours.csv`
