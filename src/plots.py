import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from pathlib import Path

from loguru import logger

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
    title: str | None = None,
    vline: float | None = None,
    ylabel: str | None = None,
    ylim: float | None = None,
):
    for run_group, label in zip(run_names, labels):
        mean_values = get_mean_metric(
            run_names=run_group,
            metric_name=metric_name,
            experiment_name=experiment_name,
            plot=False,
        )
        sns.lineplot(x=range(len(mean_values)), y=mean_values, label=label)
    plt.title(title if title else f"Mean {metric_name}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlabel("Number of marginalization images")
    plt.ylabel(ylabel or metric_name)
    if vline is not None:
        plt.axvline(x=vline, ymin=0, color="black", linestyle="--", alpha=0.7)

    if ylim is not None:
        plt.ylim(0, ylim)

    plt.legend()
    plt.show()


def plot_mean_for_nfes(
    inds: list[int],
    nfes: list[int],
    metric_name: str,
    experiment_name: str,
    title: str | None = None,
    vline: float | None = None,
    ylabel: str | None = None,
    ylim: float | None = None,   
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
        title=title,
        vline=vline,
        ylabel=ylabel,
        ylim=ylim,
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
    title: str | None = None,
    ylim: float | None = None,
) -> None:
    data = []

    for run_group, label in zip(run_names, labels):
        for run_name, target_idx in run_group:
            run = get_run_by_name(run_name, experiment_name=experiment_name)[0]

            history = {}

            for metric in metrics:
                m_hist = client.get_metric_history(run.info.run_id, metric)
                history[metric] = [m.value for m in m_hist]

            df = pd.DataFrame(history)

            ranks = df.rank(axis=1, ascending=False, method="min")
            final_rank = ranks.iloc[-1]
            is_different = (ranks != final_rank).any(axis=1)

            if not is_different.any():
                converged_step = 1
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

    if title:
        plt.title(title)
    else:
        plt.title("Feature Ranking Convergence")
    plt.xlabel("Target Image Index")
    plt.ylabel("Convergence Step")

    if ylim is not None:
        plt.ylim(0, ylim)

    plt.legend(title="Method")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_lpips_scatter_for_runs(
    run_names: list[list[str]],
    labels: list[str],
    experiment_name: str,
) -> None:
    plt.figure(figsize=(10, 8))

    for run_group, label in zip(run_names, labels):
        group_lpips = []
        group_preds = []

        for run_name in run_group:
            run = get_run_by_name(run_name, experiment_name=experiment_name)[0]

            artifacts = client.list_artifacts(run.info.run_id, "lpips")
            json_artifact = next(
                (a for a in artifacts if a.path.endswith(".json")), None
            )

            if json_artifact:
                local_path = client.download_artifacts(
                    run.info.run_id, json_artifact.path
                )
                with open(local_path, "r") as f:
                    data = json.load(f)
                    group_lpips.extend(data.get("lpips", []))
                    group_preds.extend(data.get("preds", []))
            else:
                logger.warning(f"No LPIPS JSON artifact found for run '{run_name}'.")

        plt.scatter(group_lpips, group_preds, label=label, alpha=0.5, s=20)

    plt.title("LPIPS Distance vs. Prediction Difference")
    plt.xlabel("LPIPS Distance to Target Image")
    plt.ylabel("Prediction Difference (logits)")
    plt.legend(title="Method")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_roar(
    max_top_k: int,
    metric_name: str,
    experiment_names: list[str] = ["retrain"],
    run_names: list[str] = ["i2sb_tau_0.5_topk_X"],
    labels: list[str] | None = None,
    mode: str = "boxplot",
    ylabel: str | None = None,
) -> None:
    data_rows = []
    assert len(experiment_names) == len(run_names), "experiment_names and run_names must have the same length."

    for experiment_name, run_name_template, label in zip(experiment_names, run_names, labels or run_names):
        for top_k in range(1, max_top_k + 1):
            current_run_name = run_name_template.replace("X", str(top_k))

            runs = get_run_by_name(
                current_run_name,
                experiment_name=experiment_name,
                return_multiple=True,
            )

            for run in runs:
                last_test_metric = [
                    m.value
                    for m in client.get_metric_history(run.info.run_id, metric_name)
                ][-1]

                data_rows.append(
                    {
                        "Top-k": top_k,
                        metric_name: last_test_metric,
                        "Method": label,
                    }
                )

    df = pd.DataFrame(data_rows)

    plt.figure(figsize=(14, 7))
    if mode == "boxplot":
        sns.boxplot(
            data=df,
            x="Top-k",
            y=metric_name,
            hue="Method",
            palette="viridis",
        )
    elif mode == "violin":
        sns.violinplot(
            data=df,
            x="Top-k",
            y=metric_name,
            hue="Method",
            palette="viridis",
        )
    elif mode == "lineplot":
        sns.lineplot(
            data=df,
            x="Top-k",
            y=metric_name,
            hue="Method",
            # palette="viridis",
            marker="o",
            err_style="band",
            errorbar=("ci", 95),
        )
    plt.title("ROAR Performance")
    plt.xlabel("Top-k features removed")
    plt.xticks(range(1, max_top_k + 1))
    plt.ylabel(f"Mean Test {metric_name.capitalize()}" if ylabel is None else ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_metric_for_experiment(
    experiment_names: list[str] | str,
    metric_name: str,
    labels: list[str] | None = None,
    run_names: list[list[str]] | None = None,
    mode: str = "boxplot",
) -> None:
    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    if labels is None:
        labels = experiment_names

    if len(labels) != len(experiment_names):
        raise ValueError("labels and experiment_names must have the same length.")

    data_rows = []

    for idx, (experiment_name, label) in enumerate(zip(experiment_names, labels)):
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment '{experiment_name}' not found.")
            continue

        if run_names is None:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        else:
            runs = []
            for run_name in run_names[idx]:
                runs.extend(
                    get_run_by_name(run_name, experiment_name=experiment_name)
                )

        values = []
        for run in runs:
            metric_history = client.get_metric_history(run.info.run_id, metric_name)
            values.extend(metric.value for metric in metric_history)

        if not values:
            logger.warning(
                f"No metric '{metric_name}' found in experiment '{experiment_name}'."
            )
            continue

        data_rows.extend(
            {
                "Experiment": label,
                metric_name: value,
            }
            for value in values
        )

    if not data_rows:
        logger.warning(f"No values found for metric '{metric_name}'.")
        return

    df = pd.DataFrame(data_rows)
    plt.figure(figsize=(max(6, 3 * len(labels)), 7))

    if mode == "boxplot":
        sns.boxplot(data=df, x="Experiment", y=metric_name)
    elif mode == "violin":
        sns.violinplot(data=df, x="Experiment", y=metric_name)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    plt.title(f"{metric_name} across experiments")
    plt.ylabel(metric_name)
    plt.xlabel("Experiment")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_shapley_summary_per_feature(
    experiment_1_name: str,
    experiment_2_name: str,
    run_template_1: str,
    run_template_2: str,
    indices_1: list[int],
    indices_2: list[int],
    title: str = "Shapley Summary per Feature",
) -> None:
    base_features = ["eyes", "nose", "mouth"]
    all_features = base_features + [
        "eyes + mouth",
        "eyes + nose",
        "mouth + nose"
    ]
    
    data_rows = []

    for idx in indices_1:
        run_name_1 = run_template_1.replace("{idx}", str(idx))
        run_1 = get_run_by_name(run_name_1, experiment_name=experiment_1_name)[0]
        for feature in base_features:
            history = client.get_metric_history(run_1.info.run_id, feature)
            last_val = history[-1].value
            data_rows.append({
                "Feature": feature,
                "Shapley Value": last_val,
                "Target Image": idx
            })


    for idx in indices_2:
        run_name_2 = run_template_2.replace("{idx}", str(idx))
        run_2 = get_run_by_name(run_name_2, experiment_name=experiment_2_name)[0]
        
        run_data = client.get_run(run_2.info.run_id).data
        metrics = run_data.metrics
        
        for metric_key, _ in metrics.items():
            if "_" in metric_key:
                parts = metric_key.split("_")
                normalized_key = " + ".join(sorted(parts))
                
                history = client.get_metric_history(run_2.info.run_id, metric_key)
                last_val = history[-1].value
                data_rows.append({
                    "Feature": normalized_key,
                    "Shapley Value": last_val,
                    "Target Image": idx
                })

    df = pd.DataFrame(data_rows)
    df["Feature"] = pd.Categorical(df["Feature"], categories=all_features, ordered=True)

    plt.figure(figsize=(9, 6))
    
    sns.swarmplot(
        data=df,
        x="Shapley Value",
        y="Feature",
        color="#1f77b4",
        alpha=0.6,
        size=4
    )

    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    plt.title(title)
    plt.xlabel("Shapley value", fontsize=12)
    plt.ylabel("")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # INDS = [2471, 1586, 1275, 2646, 2712, 1777, 503]#, 280, 664, 1777, 580, 503]
    INDS = [1586, 1275, 2646, 2712, 1050]
    # INDS = [2471, 1586, 1275, 2646, 2712]
    NFE = [10, 20, 50, 100]

    # plot_mean_for_nfes(
    #     inds=INDS,
    #     nfes=NFE,
    #     metric_name="max_abs_shapley_difference",
    #     experiment_name="shapley",
    #     title="Substitution + I2SB (tau=0.5)",
    #     vline=20.0,
    #     ylabel="Max Abs Shapley Difference",
    #     ylim=0.35,
    # )

    RUN_NAMES = [
        # [f"target_{ind}_male_N1_tau_0.5_nfe_10" for ind in INDS],
        # [f"target_{ind}_male_N1_tau_0.5_nfe_20" for ind in INDS],
        # [f"target_{ind}_male_N1_tau_0.5_nfe_50" for ind in INDS],
        # [f"target_{ind}_male_N1_tau_0.5_nfe_100" for ind in INDS],
        # [f"grid_search_fill_target_{ind}" for ind in INDS],
        # [f"grid_search_sub_target_{ind}_fixed" for ind in INDS],
        # [f"grid_search_i2sb_target_{ind}_tau_1.0_nfe_20" for ind in INDS],
        # [f"custom"],
        # [f"grid_search_i2sb_target_{ind}_tau_1.0_nfe_100" for ind in INDS],
        [f"lpips_target_{ind}_fill" for ind in INDS],
        [f"lpips_target_{ind}_sub" for ind in INDS],
        [f"lpips_target_{ind}_i2sb_tau_0.5" for ind in INDS],
        [f"lpips_target_{ind}_i2sb_tau_1.0" for ind in INDS]
    ]
    # LABELS = ["I2SB (NFE=10)", "I2SB (NFE=20)", "I2SB (NFE=50)", "I2SB (NFE=100)"]
    # LABELS = ["NFE=20", "NFE=100"]
    LABELS = ["Black fill", "Substitution", "Substitution + I2SB (tau=0.5)", "I2SB (tau=1.0)"]

    # plot_mean_for_runs(
    #     RUN_NAMES,
    #     LABELS,
    #     metric_name="max_abs_shapley_difference",
    #     experiment_name="shapley",
    #     title="I2SB (tau=1.0)",
    #     vline=20.0,
    #     ylabel="Max Abs Shapley Difference",
    #     ylim=0.35,
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
    #     title="Substitution",
    #     ylim=35,
    # )

    # plot_lpips_scatter_for_runs(
    #     run_names=RUN_NAMES,
    #     labels=LABELS,
    #     experiment_name="shapley",
    # )

    RUN_NAMES = [
        # "filip_blackfill_blackfill_topk_X",
        # "filip__i2sb_tau_0.5_topk_X",
        # "i2sb_tau_1.0_topk_X",
        # "sub_topk_X_2",
        # "sub_reset4_topk_X",
        # "sub_i2sb_tau_1.0_topk_X",
        # "i2sb_tau_1.0_reset_noabs_topk_X",
        # "i2sb_tau_1.0_reset_noabs_test_0.2_topk_X",
        # "i2sb_tau_1.0_reset_noabs_epochs_64_topk_X",
        # "i2sb_tau_1.0_reset_noabs_lr_0.001_topk_X",
        # "sub_reset_noabs_test_0.2_topk_X",
        # "sub_reset_noabs_epochs_64_topk_X",
        # "sub_reset_noabs_lr_0.001_topk_X",
        "random_topk_X",
        "filip_blackfill_big_seed_fixed_blackfill_X",
        "sub_reset_noabs_topk_X",
        "i2sb_tau_1.0_reset_noabs_2_topk_X",
        "filip_i2sb_big_seed_fixed_i2sb_tau_0.5_topk_X",
    ]

    EXPERIMENTS = [
        "retrain",
        "retrain",
        "retrain",
        "retrain",
        "retrain"
    ]

    # plot_roar(
    #     max_top_k=3,
    #     metric_name="test_accuracy",
    #     experiment_names=EXPERIMENTS,
    #     run_names=RUN_NAMES,
    #     labels=[
    #         "Random",
    #         "Black fill",
    #         "Substitution",
    #         "I2SB (tau=1.0)",
    #         "Substitution + I2SB (tau=0.5)",
    #         # "I2SB (tau=1.0, no abs)",
    #         # "I2SB (tau=1.0, no abs, 2)",
    #         # "I2SB (tau=1.0, no abs, test 0.2)",
    #         # "I2SB (tau=1.0, no abs, epochs 64)",
    #         # "I2SB (tau=1.0, no abs, lr 0.001)",
    #         # "Substitution (no abs)",
    #         # "Substitution (no abs, test 0.2)",
    #         # "Substitution (no abs, epochs 64)",
    #         # "Substitution (no abs, lr 0.001)",
    #     ],
    #     mode="lineplot",
    #     ylabel="Test Accuracy",
    # )

    # plot_roar(
    #     max_top_k=3,
    #     metric_name="test_mean_confidence",
    #     experiment_names=EXPERIMENTS,
    #     run_names=RUN_NAMES,
    #     mode="lineplot",
    # )

    # plot_roar(
    #     max_top_k=3,
    #     metric_name="test_loss",
    #     experiment_names=EXPERIMENTS,
    #     run_names=RUN_NAMES,
    #     mode="lineplot",
    # )

    IND1 = [1586, 1275, 2646, 2712, 1050, 933, 1242, 497, 2606, 1855, 429, 942, 2813, 1865, 1745, 173, 1552, 2356, 2683]
    # IND2 = [2471, 1586, 1275, 2646, 2712, 1050, 933, 1242, 497, 2606, 1855, 429, 942, 2813, 1865, 1745, 173, 1552, 2356, 2683, 2692, 622, 2217, 1258, 2189, 137, 988, 1622, 2781, 447, 1909, 575, 1982, 792, 2451, 2155, 1185, 386, 804, 2696]
    plot_metric_for_experiment(
        experiment_names=[
            # "bilinear_i2sb_tau_0.5_nfe_10",
            "bilinear",
            # "bilinear",
            # "bilinear",
            "bilinear",
        ],
        labels=[
            # "Black fill",
            # "I2SB",
            "Substitution",
            # "Substitution + I2SB (tau=0.5)",
            "I2SB (tau=1.0)",
        ],
        run_names=[
            # [f"filip_blackfill_ok_final_{ind}_blackfill" for ind in IND2],
            [f"fixed_{ind}_sub" for ind in IND1],
            # [f"filip_i2sb_ok_final_{ind}_i2sb_tau_0.5_nfe_10" for ind in IND2],
            [f"fixed_{ind}_i2sb_tau_1.0" for ind in IND1]
        ],
        metric_name="r_squared",
        mode="boxplot",
    )

    IND1 = [
        2471, 1586, 1275, 2646, 2712, 1050, 933, 1242, 497, 2606, 
        1855, 429, 942, 2813, 1865, 1745, 173, 1552, 2356, 2683, 
        2692, 622, 2217, 1258, 2189, 137, 988, 1622, 2781, 447, 
        1909, 575, 1982, 792, 2451, 2155, 1185, 386, 804, 2696, 
        1718, 228, 2049, 2021, 779, 2768, 1127, 674, 2257, 2060, 
        280, 664, 1777, 580, 503, 797, 2147, 502, 1215, 1688, 392, 
        2258, 1888, 456, 1954, 477, 1498, 419, 1310, 955, 1036, 
        1312, 227, 1136, 1466, 2290, 2812, 433, 1955, 2345, 2044,
        1311, 1349, 2385, 2316, 1424, 1648, 2809, 1582, 417, 1097,
        134, 2493, 1885, 434, 351, 2724, 237, 1935, 530
    ]

    IND2 = [
        2471, 1586, 1275, 2646, 2712, 1050, 933, 1242, 497, 2606, 1855, 429, 942, 2813, 1865, 1745, 173, 1552, 2356, 2683
    ]

    # plot_shapley_summary_per_feature(
    #     experiment_1_name="shapley",
    #     experiment_2_name="shapley",
    #     # run_template_1="dataset_target_{idx}_tau_1.0",
    #     # run_template_2="target_{idx}_male_N2_i2sb_tau_1.0_fixed3",
    #     run_template_1="dataset_sub_target_{idx}",
    #     run_template_2="target_{idx}_male_N2_sub_fixed3",
    #     indices_1=IND1,
    #     indices_2=IND2,
    #     title="Shapley Summary per Feature",
    # )