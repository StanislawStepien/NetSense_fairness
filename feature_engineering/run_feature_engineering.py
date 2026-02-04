from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fe_lib import (
    build_pure_demographics,
    build_topology_features,
    load_fe_config,
    merge_all_features,
    validate_fe_inputs,
    write_feature_engineering_report,
    write_feature_engineering_plots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature engineering steps.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("feature_engineering_config.json"),
        help="Path to the feature engineering config JSON.",
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
        help="Write an analysis report after feature engineering.",
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
        help="Write plots for distributions, correlations, and outliers.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("feature_engineering/plots"),
        help="Output directory for feature engineering plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_fe_config(args.config)
    config.ensure_output_dirs()

    missing_inputs = validate_fe_inputs(config)
    if missing_inputs:
        missing_list = "\n".join(f"- {p}" for p in missing_inputs)
        raise FileNotFoundError(
            "Missing required feature engineering inputs:\n" + missing_list
        )

    pure_demog = build_pure_demographics(config)

    if not args.skip_topology:
        topology = build_topology_features(config)
    else:
        topology = pd.read_csv(config.topology_features_path)

    if not args.survey_features.exists():
        raise FileNotFoundError(
            "Survey feature file not found. Provide --survey-features pointing to the engineered survey features CSV."
        )

    merged = merge_all_features(config, args.survey_features)

    if args.analysis:
        write_feature_engineering_report(args.analysis_output, pure_demog, topology, merged)
    if args.plots:
        write_feature_engineering_plots(args.plots_dir, pure_demog, topology, merged)


if __name__ == "__main__":
    main()
