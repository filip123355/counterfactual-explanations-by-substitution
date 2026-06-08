import mlflow 
from mlflow.entities import Run
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from src.mlflow import get_runs_by_names, client


def get_mean_metric(
        run_names: list[str], 
        metric_name: str,
        experiment_name: str,
        plot: bool = False,
) -> np.ndarray:
    runs = get_runs_by_names(run_names, experiment_name=experiment_name)
    values = np.zeros((len(client.get_metric_history(runs[0].info.run_id, metric_name))),)

    for run in runs:
        metric_history = client.get_metric_history(run.info.run_id, metric_name)
        values += np.array([m.value for m in metric_history])
    
    mean_values = values / len(runs)
    
    if plot:
        fig, ax = plt.subplots()
        plt.plot(range(len(mean_values)), mean_values)
        ax.set_title(f"Mean {metric_name}")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        plt.show()

    return mean_values


def plot_mean_for_nfes(
        inds: list[int],
        nfes: list[int],
        metric_name: str,
        experiment_name: str,
) -> None:
    for nfe in nfes:
        run_names = [
            f"target_{ind}_male_N1_tau_0.5_nfe_{nfe}" for ind in inds
        ]
        mean_values = get_mean_metric(
            run_names=run_names,
            metric_name=metric_name,
            experiment_name=experiment_name,
            plot=False,
        )
        plt.plot(mean_values, label=f"NFE={nfe}")
    plt.title(f"Mean {metric_name}")
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


def plot_ranking_change(
        inds: list[int],
        nfes: list[int],
        metrics: list[str],
        experiment_name: str,
) -> None:
    
    fig, ax = plt.subplots(1, len(nfes), figsize=(5 * len(nfes), 5))
    
    for i, nfe in enumerate(nfes):
        run_names = [
            f"target_{ind}_male_N1_tau_0.5_nfe_{nfe}" for ind in inds
        ]

        mean_metrics = {}
        
        for metric in metrics:
            mean_metrics[metric] = get_mean_metric(
                run_names=run_names,
                metric_name=metric,
                experiment_name=experiment_name,
                plot=False,
            )

        metric_df = pd.DataFrame(mean_metrics)
        ranks = metric_df.rank(axis=1, ascending=False, method='min')

        for metric in metrics:
            ax[i].plot(
                ranks.index,
                ranks[metric],
                label=metric,
                marker='o',
            )
            ax[i].set_title(f"Ranking of {metric} (NFE={nfe})")
            ax[i].set_xlabel("Step")
    
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__": 
    INDS = [
        2471, 1586, 1275, 2646, 2712, 280, 664, 1777, 580, 503
    ]
    NFE =[20, 50] 
    
    plot_mean_for_nfes(
        inds=INDS,
        nfes=NFE,
        metric_name="max_abs_shapley_difference",
        experiment_name="shapley",
    )

    plot_ranking_change(
        inds=INDS,
        nfes=NFE,
        metrics=["eyes", "nose", "mouth"],
        experiment_name="shapley",
    )