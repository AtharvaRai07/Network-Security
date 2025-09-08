from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataValidationArtifact
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file
from networksecurity.logging.logger import logging
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import os, sys


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self,df:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(df.columns)}")
            if len(df.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_numerical_columns(self,df:pd.DataFrame)->bool:
        try:
            number_of_numerical_columns = len(self._schema_config['numerical_columns'])
            logging.info(f"Required number of numerical columns: {number_of_numerical_columns}")
            logging.info(f"Dataframe has numerical columns: {len(df.select_dtypes(include=[np.number]).columns)}")
            if len(df.select_dtypes(include=[np.number]).columns) == number_of_numerical_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def detect_data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,threshold=0.05)->bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                if base_df[column].dtype.kind not in 'biufc':
                    continue
                logging.info(f"Detecting data drift for column: {column}")
                d1 = base_df[column]
                d2 = current_df[column]
                same_distribution = ks_2samp(d1, d2)
                if same_distribution.pvalue <= threshold:
                    # Drift detected
                    status = False
                    report.update({
                        column: {
                            "p_value": float(same_distribution.pvalue),
                            "drift_status": True
                        }
                    })
                else:
                    # No drift
                    report.update({
                        column: {
                            "p_value": float(same_distribution.pvalue),
                            "drift_status": False
                        }
                    })
            logging.info(f"Data drift detection completed. Drift status: {status}")
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Writing drift report to {drift_report_file_path}")

            write_yaml_file(file_path=drift_report_file_path,content=report)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Reading training and test data")

            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            logging.info("Validating number of columns in training data")
            status = self.validate_number_of_columns(train_df)
            if not status:
                error_message = f"Training data does not have the required number of columns"

            logging.info(f"Validating number of columns in test data")
            status = self.validate_number_of_columns(test_df)
            if not status:
                error_message = f"Test data does not have the required number of columns"

            logging.info(f"Validating number of numerical columns in training data")
            status = self.validate_number_of_numerical_columns(train_df)
            if not status:
                error_message = f"Training data does not have the required number of numerical columns"

            logging.info(f"Validating number of numerical columns in test data")
            status = self.validate_number_of_numerical_columns(test_df)
            if not status:
                error_message = f"Test data does not have the required number of numerical columns"

            logging.info(f"Checking if training and test data has drift")
            status = self.detect_data_drift(base_df=train_df,current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)


