#!/home/pekandolf/.local/bin/pdm run
import logging
from pathlib import Path
from omegaconf import OmegaConf
from importlib import import_module
from dvclive import Live
from clean_repo import require_clean_git
import model
from data import load_cats_vs_dogs
from myio import save_skops as save

def param_from_yaml(live, component):
    """Log parameters dynamically from the configuration."""
    prefix = component.type.split(".")[-1]
    for name, value in component.init_args.items():
        live.log_param(prefix + "/" + name, value)

@require_clean_git
def main():
    logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s', level=logging.DEBUG)

    # Load dataset
    X_train, y_train, X_test, y_test = load_cats_vs_dogs()

    logging.debug("Load config")
    params_path = Path(".") / "params.yaml"
    params = OmegaConf.load(params_path)

    # Dynamically load PCA class
    pca_module = import_module(params["PCA"].type.rsplit(".", 1)[0])
    PCA = getattr(pca_module, params["PCA"].type.rsplit(".", 1)[-1])
    pca = PCA(**params["PCA"].init_args)

    # Dynamically load the estimators
    estimators = []
    for estimator_params in params["VotingClassifier"].estimators:
        estimator_module = import_module(estimator_params.type.rsplit(".", 1)[0])
        EstimatorClass = getattr(estimator_module, estimator_params.type.rsplit(".", 1)[-1])
        estimator = EstimatorClass(**estimator_params.init_args)
        estimators.append(estimator)

    # Create the classifier
    voting_clf = model.get(params["VotingClassifier"], estimators)

    # Start training and logging with DVCLive
    with Live() as live:
        # Log the PCA and estimator parameters
        param_from_yaml(live, params["PCA"])
        for est in estimators:
            param_from_yaml(live, est)
        
        logging.debug("Train classifier")
        voting_clf.fit(X_train, y_train)

        # Log metrics
        metrics = model.evaluate(voting_clf, [[X_train, y_train, "train"], [X_test, y_test, "test"]])
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)

        # Predict and log confusion matrix
        y_pred = voting_clf.predict(X_test)
        live.log_sklearn_plot("confusion_matrix", y_test, y_pred)

        # Save the model
        artifacts_dir = Path(live.dir) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        file = artifacts_dir / "model.skops"
        save(voting_clf, file, X_train[:1])
        live.log_artifact(file.relative_to(Path(".")), type="model")

if __name__ == "__main__":
    main()
