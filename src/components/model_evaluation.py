import os
#import mlflow
#import mlflow.sklearn
import pandas as pd
from dvclive import Live
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.common import load_data, load_bin
#from pathlib import Path
from sklearn.metrics import (average_precision_score,
                             roc_auc_score
                             )
#from urllib.parse import urlparse
import matplotlib.pyplot as plt


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = load_bin(self.config.model_path)
        self.live = Live(os.path.join(self.config.root_dir, "live"), dvcyaml=False)
        self.data = load_data(self.config.test_data_path)

    
    def eval_metrics(self, actual, pred):
        avg_prec = average_precision_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        return avg_prec, roc_auc


    def log_into_dvc(self, data_category: str, file: str):
        data = load_data(file)
        data_x = data.drop(self.config.target_column, axis=1)
        data_y = data[self.config.target_column]

        # Evaluate train and test datasets.
        predictions_by_class = self.model.predict_proba(data_x)
        predictions = predictions_by_class[:, 1]

        y_pred = self.model.predict(data_x)

        # Using dvclive to log metrics...
        avg_prec, roc_auc = self.eval_metrics(data_y, predictions)

        if not self.live.summary:
            self.live.summary = {"avg_prec": {}, "roc_auc": {}}
        self.live.summary["avg_prec"][data_category] = avg_prec
        self.live.summary["roc_auc"][data_category] = roc_auc

        # ROC AUC plot
        self.live.log_sklearn_plot("roc", data_y, predictions, name=f"roc/{data_category}")

        # Confusion matrix plot
        self.live.log_sklearn_plot("confusion_matrix",
                            data_y.squeeze(),
                            predictions_by_class.argmax(-1),
                            name=f"cm/{data_category}"
                            )
        return ""


    def plot_feature_importance(self):
        # Dump feature importance image and show it with plots.
        fig, axes = plt.subplots(dpi=100)
        fig.subplots_adjust(bottom=0.2, top=0.95)
        importances = self.model.feature_importances_

        X = self.data.drop('quality', axis=1)
        feature_names = [f"feature {i}" for i in range(X.shape[1])]

        forest_importances = pd.Series(importances, index=feature_names).nlargest(n=30)

        axes.set_ylabel("Mean decrease in impurity")
        forest_importances.plot.bar(ax=axes)

        fig.savefig(os.path.join(self.config.root_dir, "importance.png"))


"""
    def log_into_mlflow(self):
        # For test data
        test_data = load_data(self.config.test_data_path)
        test_x = test_data.drop(self.config.target_column, axis=1)
        test_y = test_data[self.config.target_column]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = self.model.predict(test_x)
            (avg_prec, roc_auc) = self.eval_metrics(test_y, predicted_qualities)
            # Saving metrics as local
            scores = {"avg_prec": avg_prec, "roc_auc": roc_auc}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("avg_prec", avg_prec)
            mlflow.log_metric("roc_auc", roc_auc)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(self.model, "model", registered_model_name="Wine_rf_Model")
            else:
                mlflow.sklearn.log_model(self.model, "model")
"""