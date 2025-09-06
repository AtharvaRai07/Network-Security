import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where()

class NetWorkDataExtract():
    def __init__(self,database,mongo_client,collection):
        try:
            self.database = database
            self.mongo_client = pymongo.MongoClient(mongo_client,tlsCAFile=ca)
            self.collection = collection
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def cv_to_json_converter(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self,records):
        try:
            self.records = records
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    FILE_PATH = "Network_Data\phishingData.csv"
    DATABASE = "AtharvaAI"
    COLLECTION = "NetworkData"

    network_obj = NetWorkDataExtract(database=DATABASE,mongo_client=MONGO_DB_URL,collection=COLLECTION)
    records = network_obj.cv_to_json_converter(file_path=FILE_PATH)
    no_of_records = network_obj.insert_data_mongodb(records=records)
    print(no_of_records)
