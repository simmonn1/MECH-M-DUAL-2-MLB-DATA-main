#!/home/pekandolf/.local/bin/pdm run
import logging
from pathlib import Path
from omegaconf import OmegaConf
from importlib import import_module
from dvclive import Live
from clean_repo import require_clean_git
from data import load_cats_vs_dogs
from myio import save_skops as save
import model


def param_from_yaml(live, component, name_prefix=None):
    prefix = name_prefix or component.type.split(".")[-1]
    for name, value in component.init_args.items():
        live.log_param(f"{prefix}/{name}", value)


def load_component(component_conf):
    module_path, class_name = component_conf["type"].rsplit(".", 1)
    module = import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**component_conf["init_args"])


@require_clean_git
def main():
    logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s',
                        level=logging.DEBUG)

    X_train, y_train, X_test, y_test = load_cats_vs_dogs()

    logging.debug("Load config")
    params_path = Path(".") / "params.yaml"
    params = OmegaConf.load(params_path)

    # Load PCA component
    pca_conf = params["PCA"]
    pca = load_component(pca_conf)

    # Load estimators
    estimator_names = params["VotingClassifier"]["estimators"]
    estimators = []
    for name in estimator_names:
        est_conf = params[name]
        est_instance = load_component(est_conf)
        estimators.append((name, est_instance))

    # Load VotingClassifier dynamically
    voting_conf = params["VotingClassifier"]
    voting_args = dict(voting_conf["init_args"])
    voting_args["estimators"] = estimators

    voting_module, voting_class_name = voting_conf["type"].rsplit(".", 1)
    VotingClass = getattr(import_module(voting_module), voting_class_name)
    voting_clf = VotingClass(**voting_args)

    # Logging and training
    with Live() as live:
        param_from_yaml(live, pca_conf, name_prefix="PCA")
        for name in estimator_names:
            param_from_yaml(live, params[name], name_prefix=name)

        logging.debug("Train classifier")
        voting_clf.fit(X_train, y_train)

        metrics = model.evaluate(voting_clf, [[X_train, y_train, "train"],
                                              [X_test, y_test, "test"]])
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)

        y_pred = voting_clf.predict(X_test)
        live.log_sklearn_plot("confusion_matrix", y_test, y_pred)

        dir = Path(live.dir) / "artifacts"
        dir.mkdir(parents=True, exist_ok=True)
        file = dir / "model.skops"
        save(voting_clf, file, X_train[:1])


if __name__ == "__main__":
    main()
