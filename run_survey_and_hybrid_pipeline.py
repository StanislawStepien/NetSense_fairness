from __future__ import annotations

import argparse
from pathlib import Path

from feature_engineering.fe_lib import (
    build_pure_demographics,
    build_topology_features,
    load_fe_config,
    merge_all_features,
    validate_fe_inputs,
)
from survey_and_hybrid_approach_pipeline.pipeline_lib import (
    load_config,
    load_feature_tables,
    load_minorities,
    load_simulation_results,
    train_and_evaluate,
    validate_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature engineering + training end-to-end.")
    parser.add_argument(
        "--fe-config",
        type=Path,
        default=Path("feature_engineering/feature_engineering_config.json"),
        help="Path to feature engineering config JSON.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("survey_and_hybrid_approach_pipeline/config.json"),
        help="Path to training config JSON.",
    )
    parser.add_argument(
        "--survey-features",
        type=Path,
        default=Path("survey_and_hybrid_approach_pipeline/all_features_plus_neighbours.csv"),
        help="Path to the precomputed survey features (used to merge with topology).",
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="Skip recomputing topology features if they already exist.",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Write a feature engineering analysis report.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=Path("feature_engineering/feature_engineering_report.md"),
        help="Output path for the feature engineering report.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write feature engineering plots for distributions, correlations, and outliers.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("feature_engineering/plots"),
        help="Output directory for feature engineering plots.",
    )
    return parser.parse_args()


def run_feature_engineering(
    fe_config_path: Path,
    survey_features: Path,
    skip_topology: bool,
    analysis: bool,
    analysis_output: Path,
    plots: bool,
    plots_dir: Path,
) -> None:
    fe_config = load_fe_config(fe_config_path)
    fe_config.ensure_output_dirs()

    missing_inputs = validate_fe_inputs(fe_config)
    if missing_inputs:
        missing_list = "\n".join(f"- {p}" for p in missing_inputs)
        raise FileNotFoundError(
            "Missing required feature engineering inputs:\n" + missing_list
        )

    pure_demog = build_pure_demographics(fe_config)
    if not skip_topology:
        topology = build_topology_features(fe_config)
    else:
        import pandas as pd
        topology = pd.read_csv(fe_config.topology_features_path)

    if not survey_features.exists():
        raise FileNotFoundError(
            "Survey feature file not found. Provide --survey-features pointing to the engineered survey features CSV."
        )

    merged = merge_all_features(fe_config, survey_features)

    if analysis:
        from feature_engineering.fe_lib import write_feature_engineering_report
        write_feature_engineering_report(analysis_output, pure_demog, topology, merged)
    if plots:
        from feature_engineering.fe_lib import write_feature_engineering_plots
        write_feature_engineering_plots(plots_dir, pure_demog, topology, merged)


def run_training(train_config_path: Path) -> None:
    config = load_config(train_config_path)

    missing_inputs = validate_inputs(config)
    if missing_inputs:
        missing_list = "\n".join(f"- {p}" for p in missing_inputs)
        raise FileNotFoundError(
            "Missing required training inputs. Fix the following paths and re-run:\n" + missing_list
        )

    demog_df = load_feature_tables(config)
    data_frames_by_topic = load_simulation_results(config)
    minorities = load_minorities(config.minorities_pickle_path)

    train_and_evaluate(config, demog_df, data_frames_by_topic, minorities)


def main() -> None:
    args = parse_args()
    run_feature_engineering(
        args.fe_config,
        args.survey_features,
        args.skip_topology,
        args.analysis,
        args.analysis_output,
        args.plots,
        args.plots_dir,
    )
    run_training(args.train_config)


if __name__ == "__main__":
    main()
