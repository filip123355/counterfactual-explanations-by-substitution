from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
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

def export_json_artifacts_by_run_names(
    experiment_name: str,
    run_names: list[str],
    artifact_path: str,
    output_dir: str | Path,
) -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    output_base_dir = Path(output_dir)

    for run_name in run_names:
        filter_string = f"tags.mlflow.runName = '{run_name}'"
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
        )

        if not runs:
            print(f"Warning: Run '{run_name}' not found. Skipping.")
            continue

        run = runs[0]
        run_id = run.info.run_id

        run_output_dir = output_base_dir
        run_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            artifacts = client.list_artifacts(run_id, path=artifact_path)
            if not artifacts:
                print(f"Warning: No artifacts found in '{artifact_path}' for run '{run_name}'.")
                continue
            
            for artifact in artifacts:
                if not artifact.is_dir and artifact.path.endswith(".json"):
                    client.download_artifacts(
                        run_id=run_id,
                        path=artifact.path,
                        dst_path=str(run_output_dir)
                    )
            
        except Exception as e:
            print(f"Could not download artifacts for run '{run_name}': {e}")

if __name__ == "__main__":
    
    EXPERIMENT_NAME = "shapley"

    # export_experiment_to_csv(
    #     experiment_name=EXPERIMENT_NAME,
    #     output_path=f"exported_experiments/{EXPERIMENT_NAME}.csv",
    # )

    export_json_artifacts_by_run_names(
        experiment_name=EXPERIMENT_NAME,
        run_names=["target_1050_male_N1_i2sb_tau_1.0_nfe_10_fixed", "target_2712_male_N1_i2sb_tau_1.0_nfe_10_fixed"],
        artifact_path="lpips",
        output_dir="exported_artifacts",
    )