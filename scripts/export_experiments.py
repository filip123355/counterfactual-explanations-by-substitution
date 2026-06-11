from pathlib import Path

import mlflow
import pandas as pd


def export_experiment_to_csv(
    experiment_name: str,
    output_path: str | Path,
) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found."
        )

    df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(
        f"Exported {len(df)} runs "
        f"from '{experiment_name}' to '{output_path}'."
    )

    return df


if __name__ == "__main__":
    
    EXPERIMENT_NAME = "bilinear_i2sb_tau_0.5_nfe_10"

    export_experiment_to_csv(
        experiment_name=EXPERIMENT_NAME,
        output_path=f"exported_experiments/{EXPERIMENT_NAME}.csv",
    )