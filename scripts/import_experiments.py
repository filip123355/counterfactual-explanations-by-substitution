from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import pandas as pd
import mlflow

from src.constants import TRACKING_URI

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _strip_prefix(column: str, prefix: str) -> str:
    return column[len(prefix):]


def import_runs_from_csv_to_mlflow(
    csv_path: str | Path,
    target_experiment_name: str,
    tracking_uri: str | None = None,
    source_author: str | None = None,
    run_name_prefix: str | None = None,
) -> None:

    csv_path = Path(csv_path)

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    df = pd.read_csv(csv_path)

    mlflow.set_experiment(target_experiment_name)

    param_cols = [c for c in df.columns if c.startswith("params.")]
    metric_cols = [c for c in df.columns if c.startswith("metrics.")]
    tag_cols = [c for c in df.columns if c.startswith("tags.")]

    for row_idx, row in df.iterrows():
        original_run_id = row.get("run_id", None)

        original_run_name = row.get("tags.mlflow.runName", None)
        if _is_missing(original_run_name):
            original_run_name = f"imported_run_{row_idx}"

        if run_name_prefix is not None:
            run_name = f"{run_name_prefix}_{original_run_name}"
        else:
            run_name = str(original_run_name)

        with mlflow.start_run(run_name=run_name):
            if not _is_missing(original_run_id):
                mlflow.set_tag("imported_from_run_id", str(original_run_id))

            mlflow.set_tag("imported_from_csv", str(csv_path))

            if source_author is not None:
                mlflow.set_tag("source_author", source_author)

            for col in param_cols:
                value = row[col]
                if _is_missing(value):
                    continue

                key = _strip_prefix(col, "params.")
                mlflow.log_param(key, str(value))

            for col in metric_cols:
                value = row[col]
                if _is_missing(value):
                    continue

                key = _strip_prefix(col, "metrics.")

                try:
                    mlflow.log_metric(key, float(value))
                except ValueError:
                    print(
                        f"Skipping non-numeric metric: "
                        f"row={row_idx}, column={col}, value={value!r}"
                    )

            for col in tag_cols:
                value = row[col]
                if _is_missing(value):
                    continue

                key = _strip_prefix(col, "tags.")

                if key == "mlflow.runName":
                    continue
                
                if key.startswith("mlflow."):
                    key = f"original_{key}"

                mlflow.set_tag(key, str(value))

    print(
        f"Imported {len(df)} runs from '{csv_path}' "
        f"to experiment '{target_experiment_name}'."
    )


if __name__ == "__main__":
    import_runs_from_csv_to_mlflow(
    csv_path="exported_experiments/retrain.csv",
    target_experiment_name="retrain_imported",
    tracking_uri=TRACKING_URI,
    source_author="filip",
    run_name_prefix="",
)