from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig, DataValidationConfig
import sys

if __name__ == "__main__":
    try:
        logging.info("Starting data ingestion")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion_obj = DataIngestion(data_ingestion_config=data_ingestion_config)

        logging.info("Initiating data ingestion")

        data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
        print(data_ingestion_artifact)

        logging.info("Completed data ingestion")

        logging.info("Starting data validation")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation_obj = DataValidation(data_validation_config=data_validation_config,
                                             data_ingestion_artifact=data_ingestion_artifact)

        logging.info("Initiating data validation")
        data_validation_artifact = data_validation_obj.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("Completed data validation")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
