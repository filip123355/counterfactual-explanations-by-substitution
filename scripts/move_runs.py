from __future__ import annotations

import argparse
import tempfile

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.constants import TRACKING_URI


def _get_experiment_id(
    client: MlflowClient,
    experiment_name_or_id: str,
) -> str:
    experiment = client.get_experiment_by_name(experiment_name_or_id)
    if experiment is not None:
        return experiment.experiment_id

    try:
        experiment = client.get_experiment(experiment_name_or_id)
    except Exception as exc:
        raise ValueError(
            f"Experiment '{experiment_name_or_id}' does not exist."
        ) from exc

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name_or_id}' does not exist."
        )

    return experiment.experiment_id


def _get_or_create_target_experiment_id(
    client: MlflowClient,
    experiment_name_or_id: str,
) -> str:
    experiment = client.get_experiment_by_name(experiment_name_or_id)
    if experiment is not None:
        return experiment.experiment_id

    try:
        experiment = client.get_experiment(experiment_name_or_id)
    except Exception:
        experiment = None

    if experiment is not None:
        return experiment.experiment_id

    return client.create_experiment(experiment_name_or_id)


def _copy_metric_history(
    client: MlflowClient,
    source_run_id: str,
    target_run_id: str,
    metric_keys: list[str],
) -> None:
    for metric_key in metric_keys:
        history = client.get_metric_history(source_run_id, metric_key)

        for metric in history:
            client.log_metric(
                run_id=target_run_id,
                key=metric.key,
                value=metric.value,
                step=metric.step,
                timestamp=metric.timestamp,
            )


def _copy_artifacts(
    client: MlflowClient,
    source_run_id: str,
    target_run_id: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_artifact_dir = client.download_artifacts(
            run_id=source_run_id,
            path="",
            dst_path=tmp_dir,
        )

        client.log_artifacts(
            run_id=target_run_id,
            local_dir=local_artifact_dir,
        )


def copy_run(
    client: MlflowClient,
    run: Run,
    source_experiment_id: str,
    target_experiment_id: str,
) -> str:
    source_run_id = run.info.run_id

    tags = dict(run.data.tags)
    run_name = tags.get("mlflow.runName")
    tags["copied_from_run_id"] = source_run_id
    tags["copied_from_experiment_id"] = source_experiment_id

    if run_name is not None:
        tags["mlflow.runName"] = run_name

    target_run = client.create_run(
        experiment_id=target_experiment_id,
        tags=tags,
        run_name=run_name,
    )
    target_run_id = target_run.info.run_id

    for key, value in run.data.params.items():
        client.log_param(target_run_id, key, value)

    _copy_metric_history(
        client=client,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        metric_keys=list(run.data.metrics.keys()),
    )

    _copy_artifacts(
        client=client,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
    )

    client.set_terminated(
        run_id=target_run_id,
        status=run.info.status,
        end_time=run.info.end_time,
    )

    return target_run_id


def move_runs_between_experiments(
    source_experiment: str,
    target_experiment: str,
) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    source_experiment_id = _get_experiment_id(client, source_experiment)
    target_experiment_id = _get_or_create_target_experiment_id(
        client,
        target_experiment,
    )

    runs = client.search_runs(
        experiment_ids=[source_experiment_id],
        max_results=50000,
    )

    if not runs:
        print(f"No runs found in experiment '{source_experiment}'.")
        return

    for run in runs:
        target_run_id = copy_run(
            client=client,
            run=run,
            source_experiment_id=source_experiment_id,
            target_experiment_id=target_experiment_id,
        )
        print(f"Copied run {run.info.run_id} -> {target_run_id}")

    print(
        f"Copied {len(runs)} runs from '{source_experiment}' "
        f"to '{target_experiment}'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy all MLflow runs from one experiment to another, "
            "including params, tags, full metric history, and artifacts."
        )
    )
    parser.add_argument("source_experiment")
    parser.add_argument("target_experiment")
    args = parser.parse_args()

    move_runs_between_experiments(
        source_experiment=args.source_experiment,
        target_experiment=args.target_experiment,
    )


if __name__ == "__main__":
    main()
