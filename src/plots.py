import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.mlflow import get_runs_by_names, client, get_run_by_name


def get_mean_metric(
    run_names: list[str],
    metric_name: str,
    experiment_name: str,
    plot: bool = False,
) -> np.ndarray:
    runs = get_runs_by_names(run_names, experiment_name=experiment_name)
    values = np.zeros(
        (len(client.get_metric_history(runs[0].info.run_id, metric_name))),
    )

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


def plot_mean_for_runs(
    run_names: list[list[str]],
    labels: list[str],
    metric_name: str,
    experiment_name: str,
):
    for run_group, label in zip(run_names, labels):
        mean_values = get_mean_metric(
            run_names=run_group,
            metric_name=metric_name,
            experiment_name=experiment_name,
            plot=False,
        )
        plt.plot(mean_values, label=label)
    plt.title(f"Mean {metric_name}")
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


def plot_mean_for_nfes(
    inds: list[int],
    nfes: list[int],
    metric_name: str,
    experiment_name: str,
) -> None:
    run_groups = []
    for nfe in nfes:
        run_names = [f"target_{ind}_male_N1_tau_0.5_nfe_{nfe}" for ind in inds]
        run_groups.append(run_names)

    plot_mean_for_runs(
        run_names=run_groups,
        labels=[f"NFE={nfe}" for nfe in nfes],
        metric_name=metric_name,
        experiment_name=experiment_name,
    )


def plot_ranking_change(
    inds: list[int],
    nfes: list[int],
    metrics: list[str],
    experiment_name: str,
) -> None:

    fig, ax = plt.subplots(1, len(nfes), figsize=(5 * len(nfes), 5))

    for i, nfe in enumerate(nfes):
        run_names = [f"target_{ind}_male_N1_tau_0.5_nfe_{nfe}" for ind in inds]

        mean_metrics = {}

        for metric in metrics:
            mean_metrics[metric] = get_mean_metric(
                run_names=run_names,
                metric_name=metric,
                experiment_name=experiment_name,
                plot=False,
            )

        metric_df = pd.DataFrame(mean_metrics)
        ranks = metric_df.rank(axis=1, ascending=False, method="min")

        for metric in metrics:
            ax[i].plot(
                ranks.index,
                ranks[metric],
                label=metric,
                marker="o",
            )
            ax[i].set_title(f"(NFE={nfe})")
            ax[i].set_xlabel("Step")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ranking_convergence_for_runs(
    run_names: list[list[tuple[str, int]]],
    labels: list[str],
    metrics: list[str],
    experiment_name: str,
) -> None:
    data = []

    for run_group, label in zip(run_names, labels):
        for run_name, target_idx in run_group:
            run = get_run_by_name(run_name, experiment_name=experiment_name)

            history = {}

            for metric in metrics:
                m_hist = client.get_metric_history(run.info.run_id, metric)
                history[metric] = [m.value for m in m_hist]

            df = pd.DataFrame(history)

            ranks = df.rank(axis=1, ascending=False, method="min")
            final_rank = ranks.iloc[-1]
            is_different = (ranks != final_rank).any(axis=1)

            if not is_different.any():
                converged_step = ranks.index[0]
            else:
                last_diff_step = is_different[is_different].index[-1]
                converged_step = last_diff_step + 1

            data.append(
                {
                    "Target Image": str(target_idx),
                    "Method": label,
                    "Convergence Step": converged_step,
                }
            )

    results_df = pd.DataFrame(data)

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=results_df,
        x="Target Image",
        y="Convergence Step",
        hue="Method",
        palette="viridis",
        edgecolor="black",
    )

    plt.title("Feature Ranking Convergence")
    plt.xlabel("Target Image Index")
    plt.ylabel("Convergence Step")
    plt.legend(title="Method")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_roar(
    max_top_k: int,
    metric_name: str,
    experiment_name: str = "retrain",
    run_name: str = "i2sb_tau_0.5_topk_X",
) -> None:
     
    metrics = {}
    for top_k in range(1, max_top_k + 1):
        run_name = run_name.replace("X", str(top_k))
        runs = get_run_by_name(run_name, experiment_name=experiment_name, return_multiple=True)
        test_metrics = []
        for run in runs:
            last_test_metric = [
                m.value for m in client.get_metric_history(run.info.run_id, metric_name)
            ][-1]
            test_metrics.append(last_test_metric)
        metrics[top_k] = test_metrics
    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=pd.DataFrame(metrics),
        palette="viridis",
    )
    plt.title("ROAR Performance")
    plt.xlabel("Top-k")
    plt.ylabel(f"Mean Test {metric_name.capitalize()}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    INDS = [2471, 1586, 1275, 2646, 2712, 280, 664, 1777, 580, 503]
    NFE = [20, 50, 100]

    # plot_mean_for_nfes(
    #     inds=INDS,
    #     nfes=NFE,
    #     metric_name="max_abs_shapley_difference",
    #     experiment_name="shapley",
    # )

    RUN_NAMES = [
        # [f"target_{ind}_male_N1_tau_0.5_nfe_10" for ind in INDS],
        [f"target_{ind}_male_N1_tau_0.5_nfe_20" for ind in INDS],
        [f"target_{ind}_male_N1_tau_0.5_nfe_50" for ind in INDS],
        [f"target_{ind}_male_N1_tau_0.5_nfe_100" for ind in INDS],
    ]
    LABELS = ["I2SB (NFE=20)", "I2SB (NFE=50)", "I2SB (NFE=100)"]

    # plot_mean_for_runs(
    #     RUN_NAMES,
    #     LABELS,
    #     metric_name="max_abs_shapley_difference",
    #     experiment_name="shapley",
    # )

    # plot_ranking_change(
    #     inds=INDS,
    #     nfes=NFE,
    #     metrics=["eyes", "nose", "mouth"],
    #     experiment_name="shapley",
    # )

    # plot_ranking_convergence_for_runs(
    #     run_names=[
    #         list(zip(run_group, INDS)) for run_group in RUN_NAMES
    #     ],
    #     labels=LABELS,
    #     metrics=["eyes", "nose", "mouth"],
    #     experiment_name="shapley",
    # )

    plot_roar(
        max_top_k=3,
        metric_name="test_accuracy",
        experiment_name="retrain",
        run_name="i2sb_tau_0.5_topk_X",
    )
