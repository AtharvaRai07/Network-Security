import os, sys
import mlflow
from sklearn.ensemble import RandomForestClassifier
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import load_object, save_object
from networksecurity.utils.main_utils.utils import save_numpy_array_data, load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_config = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self,best_model,classification_metric):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score
            accuracy_score = classification_metric.accuracy_score

            mlflow.log_metric("F1 Score", f1_score)
            mlflow.log_metric("Precision Score", precision_score)
            mlflow.log_metric("Recall Score", recall_score)
            mlflow.log_metric("Accuracy Score", accuracy_score)
            mlflow.sklearn.log_model(best_model, "model")




    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Linear Regression": LogisticRegression(verbose=1),
                "KNearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier()
            }

            params = {
                "Linear Regression": {
                    'C': [0.1, 1.0, 10, 100],
                    'solver': ['liblinear', 'saga']
                },
                "KNearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "Decision Tree": {
                    'max_depth': [3, 5, 10,15,20,25,30, None],
                    'criterion': ['gini', 'entropy'],

                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion':['gini','entropy','log_loss'],
                    'max_depth':[3,5,10,15,20,None],
                    'max_features':['sqrt','log2',None]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            logging.info(f"Model Report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with accuracy score: {best_model_score}")
            best_model = models[best_model_name]
            logging.info(f"Best Model: {best_model}")

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            self.track_mlflow(best_model,classification_train_metric)



            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            self.track_mlflow(best_model,classification_test_metric)

            preprocessor = load_object(file_path=self.data_transformation_config.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_Model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_config.transformed_train_file_path
            test_file_path = self.data_transformation_config.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

