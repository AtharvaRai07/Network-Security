from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import sys
import os
import pymongo
import pandas as pd
import numpy as np
import certifi
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig,mongo_db_url=MONGO_DB_URL):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.client = pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def export_collection_as_dataframe(self):
        """Export collection data as pandas dataframe"""
        try:
            database_name = self.data_ingestion_config.database_name
            logging.info(f"Exporting data from database: {database_name}")
            collection_name = self.data_ingestion_config.collection_name
            logging.info(f"Exporting collection: {collection_name} from database: {database_name}")
            df = pd.DataFrame(list(self.client[database_name][collection_name].find()))
            logging.info(f"Exported {df.shape[0]} rows and {df.shape[1]} columns data from collection: {collection_name}")
            if "_id" in df.columns:
                df = df.drop(columns=["_id"],axis=1)

            df.replace(to_replace="na",value=np.nan,inplace=True)
            logging.info(f"Removed _id columns from the dataframe and replaced na values with np.nan")
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            logging.info(f"Exported data into feature store at path: {feature_store_file_path}")
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
            )
            logging.info(f"Split data into train and test set")
            logging.info(f"Exited split_data_as_train_test method of DataIngestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting train and test file path")

            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path")
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            self.export_data_into_feature_store(dataframe=dataframe)
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)



