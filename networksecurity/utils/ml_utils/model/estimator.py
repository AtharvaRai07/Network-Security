from networksecurity.constants.training_pipeline import MODEL_FILE_NAME,SAVED_MODEL_DIR
import os, sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self,x):
        try:
            X_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(X_transform)
            logging.info("Prediction completed successfully")
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
