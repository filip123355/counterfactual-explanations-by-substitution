import torch
import mlflow
from loguru import logger

from src.shapley import BilinearModel
from src.data import CelebADataset, CompositeFeature, Feature
from src.inpainter.guidance.classifier import get_classifier
from src.utils import load_config, parse_args, log_config_params
from src.constants import TRACKING_URI

FEATURE_MAP = {
    "eyes": CompositeFeature.eyes,
    "nose": Feature.nose,
    "mouth": CompositeFeature.mouth,
    "hair": Feature.hair,
}


def main():

    args = parse_args()
    config = load_config(args.config)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(config["MLFLOW_EXPERIMENT_NAME"])

    run_name = config.get(
        "MLFLOW_RUN_NAME",
        f"target_{config['TARGET_INDEX']}",
    )

    with mlflow.start_run(run_name=run_name):

        log_config_params(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", str(device))

        dataset = CelebADataset(split=config["DATASET_SPLIT"])
        model = get_classifier().to(device)
        features = [FEATURE_MAP[feature] for feature in config["FEATURES"]]

        bilinear_model = BilinearModel(
            features=features, # ty: ignore
            target_idx=config["TARGET_INDEX"],
            first_order_experiment_name=config["FIRST_ORDER_EXPERIMENT_NAME"],
            second_order_experiment_name=config["SECOND_ORDER_EXPERIMENT_NAME"],
            run_name_temp=config["SHAPLEY_RUN_NAME_TEMPLATE"],
            interaction_level=config.get("INTERACTION_LEVEL", 3),
            dataset=dataset,
        )
        r_squared = bilinear_model.calculate_r_squared(
            model=model, 
            device=device,
            predict_prob=config["PRED_PROB"],
        )

        mlflow.log_metric("r_squared", r_squared)
        
        logger.info(f"Logged R-squared: {r_squared}")


if __name__ == "__main__":
    main()
