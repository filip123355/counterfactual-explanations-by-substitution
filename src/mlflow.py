import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from src.constants import TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


def get_or_create_run(
        run_name: str,
        experiment_name: str,
) -> Run:
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(experiment_name).experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )

    for r in runs:
        return r

    return client.create_run(
        experiment_id=client.get_experiment_by_name(experiment_name).experiment_id,
        tags={"mlflow.runName": run_name},
    )


def get_run_by_name(
        run_name: str,
        experiment_name: str,
) -> Run:
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(experiment_name).experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )

    if not runs:
        raise ValueError(f"No run found with name: {run_name}")

    assert len(runs) == 1, f"Multiple runs found with name: {run_name}"

    return runs[0]


def get_runs_by_names(
        run_names: list[str],
        experiment_name: str,
) -> list[Run]:
    return [get_run_by_name(run_name, experiment_name) for run_name in run_names]