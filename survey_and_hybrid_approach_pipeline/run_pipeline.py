from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_lib import (
    load_config,
    load_feature_tables,
    load_minorities,
    load_simulation_results,
    train_and_evaluate,
    validate_inputs,
    write_pipeline_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the survey + hybrid pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the pipeline config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    missing_inputs = validate_inputs(config)
    summary_path = config.artifacts_output_dir / "PIPELINE_SUMMARY.md"
    write_pipeline_summary(summary_path, missing_inputs)

    if missing_inputs:
        missing_list = "\n".join(f"- {p}" for p in missing_inputs)
        raise FileNotFoundError(
            "Missing required inputs. Fix the following paths and re-run:\n" + missing_list
        )

    demog_df = load_feature_tables(config)
    data_frames_by_topic = load_simulation_results(config)
    minorities = load_minorities(config.minorities_pickle_path)

    train_and_evaluate(config, demog_df, data_frames_by_topic, minorities)


if __name__ == "__main__":
    main()
