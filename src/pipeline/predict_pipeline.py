import os
import yaml
import sys

import torch
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.components.algorithms import WindSpeedPredictionLSTM, HumidityMLP
from src.logger import logging
from src.exception_handler import CustomException
from src.pipeline.prediction_preprocessing import PredictPreprocessing

# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)


data = pd.read_csv(r"src\pipeline\sample_data.csv")

class PredictPipeline:
    def __init__(self,
                 input_data,
                 args=args,
                 ):
        self.args = args
        self.input_data = input_data

    def data_prepration(self):
        try:
            preparing_object = PredictPreprocessing(self.input_data)
            self.input_data = preparing_object.prediction_data_consistency()
            self.input_data = preparing_object.transformations()
        except Exception as e:
            raise CustomException(e, sys)

    
    
    def predict_wind_speed(self):
        try:
            windspeed_model_path = f"{self.args['models_save_path']}/{self.args['wind_speed_model_filename']}"
            input_size = 7
            hidden_size = 68
            num_layers = 2
            windspeed_model = WindSpeedPredictionLSTM(input_dim=input_size, hidden_dim=hidden_size, output_dim=1, num_layers=num_layers, dropout=0.2)
            windspeed_model.load_state_dict(torch.load(windspeed_model_path))
            windspeed_model.eval()
            windspeed_data = torch.from_numpy(self.input_data.astype(np.float32))
            with torch.inference_mode():
                predicted_windspeed = windspeed_model(windspeed_data.unsqueeze(0))
                print("The predicted windspeed is: ", predicted_windspeed.item())
            return predicted_windspeed.item()
        except Exception as e:
            raise CustomException(e, sys)

    def predict_humidity(self):
        try:
            humidity_model_path = f"{self.args['models_save_path']}/{self.args['humidity_model_filename']}"
            input_dim = 168
            hidden_dim = 128
            output_dim = 1
            dropout = 0.2
            humidity_model = HumidityMLP(input_dim=input_dim,
                                         hidden_dim=hidden_dim,
                                         output_dim=output_dim,
                                         dropout=dropout)
            humidity_model.load_state_dict(torch.load(humidity_model_path))
            humidity_data = torch.from_numpy(self.input_data[:,3].astype(np.float32))
            humidity_model.eval()
            with torch.inference_mode():
                predicted_humidity = humidity_model(humidity_data)
            print("The predicted humidity is: ", predicted_humidity.item())
            return predicted_humidity.item() 
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_precipitation(self):
        try:
            model_path = f"{self.args['models_save_path']}/{self.args['precipitation_model_filename']}"
            randonforest_regressor = joblib.load(model_path)
            prediction_features  = self.input_data[:, 4:].reshape(1, -1)
            predicted_precipitation = randonforest_regressor.predict(prediction_features)
            print("The predicted precipation is: ", predicted_precipitation.item())
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_temperature(self):
        try:
            model_path = f"{self.args['models_save_path']}/{self.args['temperature_model_filename']}"
            xgb_regressor = xgb.Booster()
            xgb_regressor.load_model(model_path)
            prediction_features  = self.input_data[:, 2:].reshape(1, -1)
            prediction_features = xgb.DMatrix(prediction_features)
            predicted_temperature = xgb_regressor.predict(prediction_features)
            print("The predicted temperature is: ", predicted_temperature)

        except Exception as e:
            raise CustomException(e, sys)


def pipeline():     
    object = PredictPipeline(input_data=data)
    object.data_prepration()
    logging.info("Prediction pipeline started")  
    object.predict_humidity()
    object.predict_wind_speed()
    object.predict_precipitation()
    object.predict_temperature()


pipeline()