from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class FeatureEngineeringConfig:
    base_dir: Path
    survey_raw_path: Path
    network_survey_path: Path
    topology_features_path: Path
    all_features_output_path: Path
    pure_demog_output_path: Path

    def ensure_output_dirs(self) -> None:
        self.all_features_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pure_demog_output_path.parent.mkdir(parents=True, exist_ok=True)


def load_fe_config(config_path: Path) -> FeatureEngineeringConfig:
    raw = json.loads(config_path.read_text())
    base_dir = Path(raw.get("base_dir", config_path.parent)).resolve()

    def resolve_path(p: str) -> Path:
        return (base_dir / p).resolve() if not Path(p).is_absolute() else Path(p)

    return FeatureEngineeringConfig(
        base_dir=base_dir,
        survey_raw_path=resolve_path(raw["survey_raw_path"]),
        network_survey_path=resolve_path(raw["network_survey_path"]),
        topology_features_path=resolve_path(raw["topology_features_path"]),
        all_features_output_path=resolve_path(raw["all_features_output_path"]),
        pure_demog_output_path=resolve_path(raw["pure_demog_output_path"]),
    )


def validate_fe_inputs(config: FeatureEngineeringConfig) -> List[str]:
    missing = []
    for path in [
        config.survey_raw_path,
        config.network_survey_path,
        config.topology_features_path,
    ]:
        if not path.exists():
            missing.append(str(path))
    return missing


# ----- Step 2: Demographic features (pure demog) -----

def _drop_missing_data(df: pd.DataFrame, threshold: float = 15, axis: int = 0) -> pd.DataFrame:
    if axis == 0:
        missing_percentage = df.isna().mean(axis=1) * 100
    elif axis == 1:
        missing_percentage = df.isna().mean(axis=0) * 100
    else:
        raise ValueError("Invalid axis. Use 0 for rows or 1 for columns.")

    indices_to_drop = missing_percentage[missing_percentage > threshold].index
    return df.drop(index=indices_to_drop) if axis == 0 else df.drop(columns=indices_to_drop)


def _expand_and_consolidate(df: pd.DataFrame, base_columns: List[str], target_length: int) -> pd.DataFrame:
    result: Dict[str, pd.Series] = {}
    cols_to_traverse = df.columns.values
    cols_to_traverse = cols_to_traverse[cols_to_traverse != "egoid"]
    if "egoid" in df.columns:
        result["egoid"] = df["egoid"]
    for base_col in base_columns:
        matching_columns = [col for col in cols_to_traverse if col.startswith(base_col)]
        if not matching_columns:
            continue
        consolidated = df[matching_columns[0]].copy()
        for col in matching_columns[1:]:
            mask = consolidated == df[col]
            consolidated = np.where(mask, consolidated, df[col])
        for i in range(1, target_length + 1):
            col_name = f"{base_col}_{i}"
            result[col_name] = consolidated.copy()
    return pd.DataFrame(result)


def _reshape_to_survey_format(df: pd.DataFrame, base_columns: List[str], target_length: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for i in range(1, target_length + 1):
            survey_row = {f"{base_col}": row[f"{base_col}_{i}"] for base_col in base_columns}
            survey_row["SurveyNr"] = i
            survey_row["egoid"] = row["egoid"]
            rows.append(survey_row)
    return pd.DataFrame(rows)


def _safe_convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def _apply_custom_bins(df: pd.DataFrame) -> pd.DataFrame:
    custom_bins = {
        "weight": {
            "<120": lambda x: _safe_convert_to_float(x) < 120,
            "120-150": lambda x: 120 <= _safe_convert_to_float(x) <= 150,
            "150-180": lambda x: 150 < _safe_convert_to_float(x) <= 180,
            ">180": lambda x: _safe_convert_to_float(x) > 180,
        },
        "disabilitylearning": {
            "No": lambda x: str(x).lower() == "no",
            "Yes": lambda x: str(x).lower() == "yes",
            "Unknown": lambda x: pd.isna(x) or str(x).lower() == "nan",
        },
        "ethnicity": {
            "Caucasian": lambda x: "white/caucasian" in str(x).lower(),
            "Latino": lambda x: "latino" in str(x).lower() or "mexican american" in str(x).lower() or "puerto rican" in str(x).lower(),
            "Asian": lambda x: "asian" in str(x).lower(),
            "Black": lambda x: "african american" in str(x).lower(),
            "Other": lambda x: "other" in str(x).lower(),
            "Native American": lambda x: "american indian" in str(x).lower(),
        },
        "disabilityphysical": {
            "No": lambda x: str(x).lower() == "no",
            "Yes": lambda x: str(x).lower() == "yes",
            "Unknown": lambda x: pd.isna(x) or str(x).lower() == "nan",
        },
        "momrelig": {
            "Christian": lambda x: any(sub in str(x).lower() for sub in ["roman catholic", "methodist", "baptist", "presbyterian", "united church", "lutheran"]),
            "Non-Christian": lambda x: any(sub in str(x).lower() for sub in ["hindu", "buddhist", "jewish"]),
            "Agnostic/Atheist": lambda x: any(sub in str(x).lower() for sub in ["agnostic", "atheist"]),
            "Other/Unknown": lambda x: pd.isna(x) or "not sure" in str(x).lower() or "not applicable" in str(x).lower(),
        },
        "daded": {
            "High School or Less": lambda x: any(sub in str(x).lower() for sub in ["high school", "junior high"]),
            "Some College": lambda x: "some college" in str(x).lower(),
            "Graduate Degree": lambda x: "graduate degree" in str(x).lower(),
            "Unknown": lambda x: pd.isna(x) or "not sure" in str(x).lower(),
        },
        "familymilitary": {
            "Connected": lambda x: str(x).lower() == "yes",
            "Not Connected": lambda x: str(x).lower() == "no",
        },
        "heighttotal": {
            "<60": lambda x: _safe_convert_to_float(x) < 60,
            "60-65": lambda x: 60 <= _safe_convert_to_float(x) <= 65,
            "65-70": lambda x: 65 < _safe_convert_to_float(x) <= 70,
            "70+": lambda x: _safe_convert_to_float(x) > 70,
        },
        "numberpets": {
            "0": lambda x: _safe_convert_to_float(x) == 0,
            "1-2": lambda x: 1 <= _safe_convert_to_float(x) <= 2,
            "3-5": lambda x: 3 <= _safe_convert_to_float(x) <= 5,
            "6+": lambda x: _safe_convert_to_float(x) > 5,
        },
        "computeruse": {
            "<5 hours": lambda x: _safe_convert_to_float(x) < 5,
            "5-7 hours": lambda x: 5 <= _safe_convert_to_float(x) <= 7,
            "8+ hours": lambda x: _safe_convert_to_float(x) > 7,
        },
        "pincome": {
            "Very Low (<10k)": lambda x: "less than 10k" in str(x).lower(),
            "Low (<40k)": lambda x: any(inc in str(x) for inc in ["10000", "15000", "20000", "25000", "30000", "40000"]),
            "Middle (40k-100k)": lambda x: any(inc in str(x) for inc in ["50000", "60000", "75000", "99999"]),
            "High (100k-250k)": lambda x: any(inc in str(x) for inc in ["100000", "150000", "200000", "249999"]),
            "Very High (>250k)": lambda x: "250000 or more" in str(x).lower(),
            "Not Sure": lambda x: "not sure" in str(x).lower(),
            "Unknown": lambda x: pd.isna(x) or "nan" in str(x).lower(),
        },
        "parentsmarriage": {
            "Together": lambda x: "alive and living together" in str(x).lower(),
            "Divorced/Separated": lambda x: "alive, but divorced" in str(x).lower(),
            "Deceased": lambda x: "one of them deceased" in str(x).lower(),
            "Unknown": lambda x: pd.isna(x) or "nan" in str(x).lower(),
        },
        "major": {
            "Science": lambda x: any(sub in str(x).lower() for sub in ["biological sciences", "physics", "chemistry", "mathematics"]),
            "Engineering": lambda x: "engineering" in str(x).lower(),
            "Business": lambda x: "business" in str(x).lower(),
            "Arts and Humanities": lambda x: any(sub in str(x).lower() for sub in ["arts", "music", "philosophy", "english", "literature", "history"]),
            "Social Science": lambda x: any(sub in str(x).lower() for sub in ["political sciences", "psychology", "anthropology", "economics"]),
            "Professional Studies": lambda x: any(sub in str(x).lower() for sub in ["medicine", "dentistry", "veterinary", "therapy", "architecture"]),
            "Other": lambda x: "undecided" in str(x).lower() or "other" in str(x).lower(),
        },
        "fbprivacy": {
            "Public": lambda x: "public" in str(x).lower(),
            "Friends Only": lambda x: "all my friends" in str(x).lower(),
            "Some Friends": lambda x: "only some of my friends" in str(x).lower(),
            "Private": lambda x: "no one can see" in str(x).lower(),
            "Unknown": lambda x: pd.isna(x) or "nan" in str(x).lower(),
        },
        "dadrelig": {
            "Christian": lambda x: any(sub in str(x).lower() for sub in ["roman catholic", "methodist", "baptist", "presbyterian", "united church", "lutheran"]),
            "Non-Christian": lambda x: any(sub in str(x).lower() for sub in ["hindu", "buddhist", "jewish"]),
            "Agnostic/Atheist": lambda x: any(sub in str(x).lower() for sub in ["agnostic", "atheist"]),
            "Other/Unknown": lambda x: pd.isna(x) or "not sure" in str(x).lower() or "not applicable" in str(x).lower(),
        },
        "eyeglasses": {
            "Yes": lambda x: str(x).lower() == "yes",
            "No": lambda x: str(x).lower() == "no",
        },
        "momed": {
            "High School or Less": lambda x: any(sub in str(x).lower() for sub in ["high school", "junior high"]),
            "Some College": lambda x: "some college" in str(x).lower(),
            "Graduate Degree": lambda x: "graduate degree" in str(x).lower(),
            "Unknown": lambda x: pd.isna(x) or "not sure" in str(x).lower(),
        },
    }

    def bin_column_values(column_series, bins):
        def map_value(value):
            for bin_label, condition in bins.items():
                if condition(value):
                    return bin_label
            return "Other"
        return column_series.map(map_value)

    for column, bins in custom_bins.items():
        if column in df.columns:
            df[column] = bin_column_values(df[column], bins)
    return df


def build_pure_demographics(config: FeatureEngineeringConfig) -> pd.DataFrame:
    all_surveys_data = pd.read_csv(config.survey_raw_path)

    columns_to_include = [
        "computeruse",
        "contactlens",
        "eyeglasses",
        "dadrelig",
        "momrelig",
        "numberpets",
        "parentsmarriage",
        "religcateg",
        "weight",
        "major",
        "disabilitylearning",
        "gender",
        "heighttotal",
        "familymilitary",
        "fbprivacy",
        "disabilityphysical",
        "ethnicity",
        "pincome",
        "momed",
        "daded",
    ]

    all_cols = all_surveys_data.columns
    cols = [col for col in all_cols if any(col.startswith(col2) for col2 in columns_to_include)]
    cols.append("egoid")
    all_surveys_data = all_surveys_data[cols]

    cols_to_drop = [col for col in cols if col.startswith("programwhich")]
    if cols_to_drop:
        all_surveys_data = all_surveys_data.drop(columns=cols_to_drop)

    all_surveys_data = _drop_missing_data(df=all_surveys_data, threshold=20, axis=1)

    cols_with_id = all_surveys_data.columns.values
    cols_without_id = cols_with_id[cols_with_id != "egoid"]

    base_columns = sorted(set(col.split("_")[0] for col in cols_without_id))
    consolidated_df = _expand_and_consolidate(all_surveys_data, base_columns, 6)
    reshaped_df = _reshape_to_survey_format(consolidated_df, base_columns, 6)
    reshaped_df = _drop_missing_data(reshaped_df, 15, axis=0)

    numeric_columns = ["weight", "heighttotal", "numberpets", "computeruse"]
    for column in numeric_columns:
        if column in reshaped_df.columns:
            reshaped_df[column] = reshaped_df[column].apply(_safe_convert_to_float)

    binned_df = _apply_custom_bins(reshaped_df)

    # Impute categorical values with most frequent
    categorical_cols = binned_df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        binned_df[col] = binned_df[col].fillna(binned_df[col].mode(dropna=True)[0])

    # Encode categorical columns
    for col in categorical_cols:
        binned_df[col] = binned_df[col].astype("category").cat.codes

    binned_df.to_csv(config.pure_demog_output_path, index=False)
    return binned_df


# ----- Step 3: Network/topology features -----

def build_topology_features(config: FeatureEngineeringConfig) -> pd.DataFrame:
    df = pd.read_csv(config.network_survey_path)

    def transform_column_name(col_name):
        return col_name.rsplit("_", 1)[0]

    variable_columns = [col for col in df.columns if col.rsplit("_", 1)[-1].isdigit()]
    base_names = {col: transform_column_name(col) for col in variable_columns}
    constant_columns = [col for col in df.columns if col not in variable_columns]

    reshaped_data = []
    max_survey_num = max(int(col.rsplit("_", 1)[-1]) for col in variable_columns)
    for i in range(1, max_survey_num + 1):
        filtered_columns = [col for col in variable_columns if col.endswith(f"_{i}")]
        completed_col = f"Completed_{i}"

        if completed_col in df.columns:
            filtered_df = df[df[completed_col].notna()]
            if filtered_columns:
                temp_df = filtered_df[constant_columns].copy()
                temp_df.update(filtered_df[filtered_columns].rename(columns=base_names))
                temp_df["SurveyNr"] = i
                reshaped_data.append(temp_df)

    final_df = pd.concat(reshaped_data, ignore_index=True)

    aggregated_df = final_df.groupby(["EgoID", "SurveyNr"]).agg(
        neighbours=("AlterID", lambda x: list(set(x))),
        neighbours_count=("AlterID", lambda x: len(set(x))),
    ).reset_index()

    network_data = []
    for survey, group in aggregated_df.groupby("SurveyNr"):
        G = nx.Graph()
        for _, row in group.iterrows():
            ego = row["EgoID"]
            for neighbor in row["neighbours"]:
                G.add_edge(ego, neighbor)

        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        community_map = {node: idx for idx, community in enumerate(communities) for node in community}

        for ego in degree_centrality.keys():
            network_data.append(
                {
                    "EgoID": ego,
                    "SurveyNr": survey,
                    "Degree Centrality": degree_centrality[ego],
                    "Closeness Centrality": closeness_centrality[ego],
                    "Betweenness Centrality": betweenness_centrality[ego],
                    "Eigenvector Centrality": eigenvector_centrality[ego],
                    "Community": community_map.get(ego, -1),
                }
            )

    network_df = pd.DataFrame(network_data)
    network_df.to_csv(config.topology_features_path, index=False)
    return network_df


# ----- Step 4: Merge engineered survey features + topology -----

def merge_all_features(config: FeatureEngineeringConfig, survey_features_path: Path) -> pd.DataFrame:
    demog_df = pd.read_csv(survey_features_path)
    topo_df = pd.read_csv(config.topology_features_path)

    if "EgoID" in topo_df.columns and "egoid" not in topo_df.columns:
        topo_df = topo_df.rename(columns={"EgoID": "egoid"})

    merged_df = demog_df.merge(topo_df, on=["egoid", "SurveyNr"], how="left")
    merged_df = merged_df.dropna()
    merged_df.to_csv(config.all_features_output_path, index=False)
    return merged_df


# ----- Optional analysis -----
def write_feature_engineering_report(
    output_path: Path,
    pure_demog: pd.DataFrame,
    topology: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Feature Engineering Report")
    lines.append("")

    def add_section(title: str, df: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        lines.append(f"- Rows: {len(df)}")
        lines.append(f"- Columns: {len(df.columns)}")
        missing = (df.isnull().mean() * 100).sort_values(ascending=False)
        top_missing = missing[missing > 0].head(10)
        if top_missing.empty:
            lines.append("- Missing data: none")
        else:
            lines.append("- Top missing columns (percent):")
            for col, pct in top_missing.items():
                lines.append(f"- {col}: {pct:.2f}%")
        lines.append("")

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            lines.append("### Numeric Summary")
            desc = numeric_df.describe().T
            for _, row in desc.iterrows():
                lines.append(
                    f"- {row.name}: mean={row['mean']:.4f}, std={row['std']:.4f}, "
                    f"min={row['min']:.4f}, p25={row['25%']:.4f}, "
                    f"median={row['50%']:.4f}, p75={row['75%']:.4f}, max={row['max']:.4f}"
                )
            lines.append("")

            lines.append("### Top Correlations (absolute)")
            corr = numeric_df.corr().abs()
            corr.values[np.tril_indices_from(corr)] = 0
            top_pairs = corr.stack().sort_values(ascending=False).head(10)
            if top_pairs.empty:
                lines.append("- No correlations found")
            else:
                for (a, b), val in top_pairs.items():
                    lines.append(f"- {a} ↔ {b}: {val:.4f}")
            lines.append("")

            lines.append("### Outlier Counts (IQR)")
            outlier_lines = []
            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                if series.empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = series[(series < lower) | (series > upper)]
                if len(outliers) > 0:
                    outlier_lines.append(f"- {col}: {len(outliers)}")
            if not outlier_lines:
                lines.append("- No outliers detected by IQR rule")
            else:
                lines.extend(outlier_lines)
            lines.append("")

    add_section("Pure Demographics", pure_demog)
    add_section("Topology Features", topology)
    add_section("Merged Features", merged)

    output_path.write_text("\n".join(lines))


def write_feature_engineering_plots(
    output_dir: Path,
    pure_demog: pd.DataFrame,
    topology: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def plot_numeric_distributions(title: str, df: pd.DataFrame) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return
        columns = list(numeric_df.columns)[:20]
        n_cols = 4
        n_rows = int(np.ceil(len(columns) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, columns):
            sns.histplot(numeric_df[col].dropna(), kde=False, ax=ax, bins=30)
            ax.set_title(col)
        for ax in axes[len(columns):]:
            ax.axis("off")
        fig.suptitle(f"{title} - Numeric Distributions (first 20)", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"{title.lower().replace(' ', '_')}_histograms.png")
        plt.close(fig)

    def plot_correlation_heatmap(title: str, df: pd.DataFrame) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"{title} - Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(output_dir / f"{title.lower().replace(' ', '_')}_correlation_heatmap.png")
        plt.close(fig)

    def plot_outlier_boxplots(title: str, df: pd.DataFrame) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return
        columns = list(numeric_df.columns)[:20]
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(data=numeric_df[columns], ax=ax)
        ax.set_title(f"{title} - Boxplots (first 20 numeric columns)")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(output_dir / f"{title.lower().replace(' ', '_')}_boxplots.png")
        plt.close(fig)

    plot_numeric_distributions("Pure Demographics", pure_demog)
    plot_correlation_heatmap("Pure Demographics", pure_demog)
    plot_outlier_boxplots("Pure Demographics", pure_demog)

    plot_numeric_distributions("Topology Features", topology)
    plot_correlation_heatmap("Topology Features", topology)
    plot_outlier_boxplots("Topology Features", topology)

    plot_numeric_distributions("Merged Features", merged)
    plot_correlation_heatmap("Merged Features", merged)
    plot_outlier_boxplots("Merged Features", merged)
