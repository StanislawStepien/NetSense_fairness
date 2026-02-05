from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Integer


@dataclass
class PipelineConfig:
    base_dir: Path

    demog_features_path: Path
    pure_demog_path: Path
    topology_features_path: Optional[Path]

    simulation_results_dir: Path
    minorities_pickle_path: Path

    models_output_dir: Path
    evaluation_output_dir: Path
    artifacts_output_dir: Path

    topics: List[str] = field(default_factory=lambda: [
        "euthanasia",
        "fssocsec",
        "fswelfare",
        "jobguar",
        "marijuana",
        "toomucheqrights",
    ])

    test_size: float = 0.2
    random_seed: int = 42
    bayes_search_iterations: int = 30
    cv_splits: int = 10
    top_feature_counts: List[int] = field(default_factory=lambda: [15, 20, 30, 40, 50, 60, 70, 85])

    def ensure_output_dirs(self) -> None:
        self.models_output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_output_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path) -> PipelineConfig:
    raw = json.loads(config_path.read_text())
    base_dir = Path(raw.get("base_dir", config_path.parent)).resolve()

    def resolve_path(p: Optional[str]) -> Optional[Path]:
        if p is None:
            return None
        return (base_dir / p).resolve() if not Path(p).is_absolute() else Path(p)

    return PipelineConfig(
        base_dir=base_dir,
        demog_features_path=resolve_path(raw["demog_features_path"]),
        pure_demog_path=resolve_path(raw["pure_demog_path"]),
        topology_features_path=resolve_path(raw.get("topology_features_path")),
        simulation_results_dir=resolve_path(raw["simulation_results_dir"]),
        minorities_pickle_path=resolve_path(raw["minorities_pickle_path"]),
        models_output_dir=resolve_path(raw["models_output_dir"]),
        evaluation_output_dir=resolve_path(raw["evaluation_output_dir"]),
        artifacts_output_dir=resolve_path(raw["artifacts_output_dir"]),
        topics=raw.get("topics") or [
            "euthanasia",
            "fssocsec",
            "fswelfare",
            "jobguar",
            "marijuana",
            "toomucheqrights",
        ],
        test_size=raw.get("test_size", 0.2),
        random_seed=raw.get("random_seed", 42),
        bayes_search_iterations=raw.get("bayes_search_iterations", 30),
        cv_splits=raw.get("cv_splits", 10),
        top_feature_counts=raw.get("top_feature_counts") or [15, 20, 30, 40, 50, 60, 70, 85],
    )


def validate_inputs(config: PipelineConfig) -> List[str]:
    missing = []
    if not config.demog_features_path.exists():
        missing.append(str(config.demog_features_path))
    if not config.pure_demog_path.exists():
        missing.append(str(config.pure_demog_path))
    if config.topology_features_path and not config.topology_features_path.exists():
        missing.append(str(config.topology_features_path))
    if not config.simulation_results_dir.exists():
        missing.append(str(config.simulation_results_dir))
    if not config.minorities_pickle_path.exists():
        missing.append(str(config.minorities_pickle_path))
    return missing


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "EgoID" in df.columns and "egoid" not in df.columns:
        df = df.rename(columns={"EgoID": "egoid"})
    return df


def load_feature_tables(config: PipelineConfig) -> pd.DataFrame:
    demog_df = pd.read_csv(config.demog_features_path)
    pure_demog = pd.read_csv(config.pure_demog_path)

    demog_df = _normalize_id_columns(demog_df)
    pure_demog = _normalize_id_columns(pure_demog)

    identifier_cols = ["egoid", "SurveyNr"]

    demog_and_pure = demog_df.merge(pure_demog, how="left", on=identifier_cols)

    if config.topology_features_path and config.topology_features_path.exists():
        topo_df = pd.read_csv(config.topology_features_path)
        topo_df = _normalize_id_columns(topo_df)
        cols_to_drop = ["Column1", "Question", "coding_success"]
        existing = [c for c in cols_to_drop if c in topo_df.columns]
        if existing:
            topo_df = topo_df.drop(columns=existing)
        topo_feature_cols = [c for c in topo_df.columns if c not in identifier_cols]
        already_present = [c for c in topo_feature_cols if c in demog_and_pure.columns]
        if not already_present:
            demog_and_pure = demog_and_pure.merge(topo_df, how="left", on=identifier_cols)

    columns_to_drop = [
        "abortion",
        "premaritalsex",
        "euthanasia",
        "homosexual",
        "deathpen",
        "marijuana",
        "eqchances",
        "fssocsec",
        "toomucheqrights",
        "Unnamed: 0",
    ]
    columns_to_drop = [c for c in columns_to_drop if c in demog_and_pure.columns]
    if columns_to_drop:
        demog_and_pure = demog_and_pure.drop(columns=columns_to_drop)

    demog_and_pure = demog_and_pure.drop_duplicates()
    demog_and_pure["egoid"] = demog_and_pure["egoid"].astype("int64")
    demog_and_pure["SurveyNr"] = demog_and_pure["SurveyNr"].astype("int64")

    return demog_and_pure


def _read_simulation_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, delimiter=";")
    except Exception:
        df = pd.read_csv(path)
    df = _normalize_id_columns(df)
    if "StudentID" in df.columns and "egoid" not in df.columns:
        df = df.rename(columns={"StudentID": "egoid"})
    return df


def load_simulation_results(config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    files = sorted(config.simulation_results_dir.glob("*.csv"))
    files_by_topic: Dict[str, List[Path]] = {topic: [] for topic in config.topics}
    for file_path in files:
        for topic in config.topics:
            if topic in file_path.name:
                files_by_topic[topic].append(file_path)

    data_frames_by_topic: Dict[str, pd.DataFrame] = {}
    needed_cols = {"egoid", "SurveyNr", "OpinionSim", "OpinionSurvey"}

    for topic, topic_files in files_by_topic.items():
        frames = []
        for file_path in topic_files:
            df = _read_simulation_csv(file_path)
            available = [c for c in df.columns if c in needed_cols]
            if not available:
                continue
            df = df[available]
            frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates()
            combined["egoid"] = combined["egoid"].astype("int64")
            combined["SurveyNr"] = combined["SurveyNr"].astype("int64")
            data_frames_by_topic[topic] = combined

    return data_frames_by_topic


def load_minorities(minorities_path: Path) -> Dict[str, pd.DataFrame]:
    minorities = joblib.load(minorities_path)
    normalized: Dict[str, pd.DataFrame] = {}
    for category, value in minorities.items():
        if isinstance(value, pd.DataFrame):
            df = value.copy()
        else:
            df = pd.DataFrame(value)
        df = _normalize_id_columns(df)
        normalized[category] = df
    return normalized


def _extract_minority_ids(minorities: Dict[str, pd.DataFrame]) -> np.ndarray:
    ids = []
    for df in minorities.values():
        if "egoid" in df.columns:
            ids.append(df["egoid"].unique())
    if not ids:
        return np.array([], dtype="int64")
    return np.unique(np.concatenate(ids))


def build_topic_dataset(
    topic: str,
    demog_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    minorities: Dict[str, pd.DataFrame],
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    merged = topic_df.merge(demog_df, on=["egoid", "SurveyNr"], how="inner")
    merged = merged.drop_duplicates()
    merged["Y"] = (merged["OpinionSim"] != merged["OpinionSurvey"]).astype(int)

    minority_ids = _extract_minority_ids(minorities)
    minority_data = merged[merged["egoid"].isin(minority_ids)]
    non_minority_data = merged[~merged["egoid"].isin(minority_ids)]

    if minority_data.empty or non_minority_data.empty:
        raise ValueError(
            f"Topic {topic} has empty minority or non-minority data after merge."
        )

    minority_train, minority_test = train_test_split(
        minority_data,
        test_size=config.test_size,
        random_state=config.random_seed,
    )
    non_minority_train, non_minority_test = train_test_split(
        non_minority_data,
        test_size=config.test_size,
        random_state=config.random_seed,
    )

    train = pd.concat([minority_train, non_minority_train])
    test = pd.concat([minority_test, non_minority_test])

    columns_to_drop = ["OpinionSim", "OpinionSurvey", "egoid", "SurveyNr", "Y"]
    X_train = train.drop(columns=[c for c in columns_to_drop if c in train.columns])
    X_test = test.drop(columns=[c for c in columns_to_drop if c in test.columns])
    y_train = train["Y"].astype(int)
    y_test = test["Y"].astype(int)

    X_minorities_test = minority_test.drop(columns=["Y", "OpinionSim", "OpinionSurvey"])
    y_minorities_test = minority_test[["Y", "egoid"]]

    return X_train, X_test, y_train, y_test, X_minorities_test, y_minorities_test


def evaluate_on_minorities(
    model: RandomForestClassifier,
    features: List[str],
    X: pd.DataFrame,
    y: pd.DataFrame,
    minorities: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    results = []
    clean_features = [f for f in features if f not in ("egoid", "Y") and f in X.columns]

    for category, df in minorities.items():
        if "egoid" not in df.columns:
            results.append({"Category": category, "F1 Score": "No EgoID"})
            continue
        minority_ids = df["egoid"].unique()
        filtered_X = X[X["egoid"].isin(minority_ids)]
        filtered_y = y[y["egoid"].isin(minority_ids)]["Y"]

        if filtered_X.empty:
            results.append({"Category": category, "F1 Score": "No Data"})
            continue

        filtered_X = filtered_X[clean_features]
        predictions = model.predict(filtered_X)
        f1 = f1_score(filtered_y, predictions, average="macro")
        results.append({"Category": category, "F1 Score": f1})

    return pd.DataFrame(results)


def train_and_evaluate(
    config: PipelineConfig,
    demog_df: pd.DataFrame,
    data_frames_by_topic: Dict[str, pd.DataFrame],
    minorities: Dict[str, pd.DataFrame],
) -> Dict[str, dict]:
    config.ensure_output_dirs()

    rf_param_grid = {
        "n_estimators": Integer(50, 400),
        "max_features": ["sqrt", "log2"],
        "max_depth": (2, 8),
        "min_samples_split": (4, 8),
        "min_samples_leaf": (2, 4),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
    }

    best_for_topic: Dict[str, dict] = {}

    for topic in config.topics:
        if topic not in data_frames_by_topic:
            best_for_topic[topic] = {"error": "No simulation data for topic"}
            continue

        X_train, X_test, y_train, y_test, X_minor, y_minor = build_topic_dataset(
            topic, demog_df, data_frames_by_topic[topic], minorities, config
        )

        is_hybrid = config.topology_features_path is not None

        model_candidates = []
        if is_hybrid:
            model_candidates = [
                ("RandomForestClassifier_balanced", RandomForestClassifier(class_weight="balanced", random_state=config.random_seed)),
                ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=config.random_seed)),
            ]
        else:
            model_candidates = [
                ("RandomForestClassifier", RandomForestClassifier(random_state=config.random_seed)),
                ("RandomForestClassifier_balanced", RandomForestClassifier(class_weight="balanced", random_state=config.random_seed)),
                ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=config.random_seed)),
            ]

        model_comparison = []
        best_baseline_name = None
        best_baseline_model = None
        best_baseline_f1 = -1.0

        for name, model in model_candidates:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds)
            model_comparison.append({"model": name, "f1": f1})
            if f1 > best_baseline_f1:
                best_baseline_f1 = f1
                best_baseline_name = name
                best_baseline_model = model

        if best_baseline_model is None:
            raise RuntimeError(f"No baseline model could be trained for topic {topic}.")

        tuned_model = best_baseline_model
        best_params = {}

        if isinstance(best_baseline_model, RandomForestClassifier):
            cv = StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_seed)
            scoring_metric = make_scorer(f1_score)

            grid_search = BayesSearchCV(
                estimator=best_baseline_model,
                search_spaces=rf_param_grid,
                cv=cv,
                scoring=scoring_metric,
                n_jobs=-1,
                verbose=0,
                n_iter=config.bayes_search_iterations,
            )
            grid_search.fit(X_train, y_train)
            tuned_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        feature_importances = getattr(tuned_model, "feature_importances_", None)
        if feature_importances is None:
            raise RuntimeError(f"Selected model for topic {topic} has no feature importances.")

        sorted_features = [x for _, x in sorted(zip(feature_importances, X_train.columns), reverse=True)]

        best_f1 = -1.0
        best_features_subset: List[str] = []
        best_model_final: Optional[RandomForestClassifier] = None
        results = []

        for no_features in config.top_feature_counts:
            selected_features = sorted_features[:no_features]
            if isinstance(tuned_model, RandomForestClassifier):
                model = RandomForestClassifier(
                    **best_params,
                    random_state=config.random_seed,
                    class_weight=tuned_model.class_weight,
                )
            else:
                model = DecisionTreeClassifier(random_state=config.random_seed)
            model.fit(X_train[selected_features], y_train)
            y_pred = model.predict(X_test[selected_features])
            f1 = f1_score(y_test, y_pred)
            results.append({"Top Features": no_features, "F1 Score": f1})

            if f1 > best_f1:
                best_f1 = f1
                best_features_subset = selected_features
                best_model_final = model

        if best_model_final is None:
            raise RuntimeError(f"No model selected for topic {topic}.")

        model_path = config.models_output_dir / f"{topic}_best_model_for_selected_features.joblib"
        joblib.dump(best_model_final, model_path)

        minority_results = evaluate_on_minorities(
            best_model_final,
            best_features_subset,
            X_minor,
            y_minor,
            minorities,
        )

        eval_path = config.evaluation_output_dir / f"evaluation_results_{topic}.csv"
        minority_results.to_csv(eval_path, index=False)

        best_for_topic[topic] = {
            "results": results,
            "best_f1": best_f1,
            "best_features_subset": best_features_subset,
            "best_model_path": str(model_path),
            "best_params": best_params,
            "model_comparison": model_comparison,
            "selected_baseline_model": best_baseline_name,
            "minorities_results": minority_results.to_dict(orient="records"),
        }

    joblib.dump(best_for_topic, config.artifacts_output_dir / "best_for_topic.joblib")
    (config.artifacts_output_dir / "best_for_topic.json").write_text(
        json.dumps(best_for_topic, indent=2)
    )

    return best_for_topic


def write_pipeline_summary(output_path: Path, missing_inputs: List[str]) -> None:
    lines = []
    lines.append("# Survey + Hybrid Pipeline Summary")
    lines.append("")

    if missing_inputs:
        lines.append("## Missing Inputs")
        for item in missing_inputs:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Pipeline Steps")
    lines.append("- Load demographic features and pure demographic features")
    lines.append("- Optionally merge topology-based features if provided")
    lines.append("- Drop opinion target columns from the feature matrix")
    lines.append("- Load simulation results per topic and normalize identifiers")
    lines.append("- Merge simulation outcomes with features and create label Y")
    lines.append("- Build balanced train/test splits for minority and non-minority groups")
    lines.append("- Run Bayesian hyperparameter search for RandomForest")
    lines.append("- Evaluate top-K feature subsets and choose best")
    lines.append("- Evaluate minority performance by group and save results")
    lines.append("- Save best models and artifacts per topic")

    output_path.write_text("\n".join(lines))
