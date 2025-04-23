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
from src.pipeline.prediction_preprocessing import prediction_preprocessing_pipeline

# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

scaler = joblib.load(args["scalar_path"])


class PredictPipeline:
    def __init__(self,
                 args,
                 data=None
                 ):
        
        self.args = args
        self.data = data

    def data_prepration(self):
        try:
            self.data = prediction_preprocessing_pipeline()
            return self.data
        except Exception as e:
            raise CustomException(e, sys) 
    
    def predict_wind_speed(self):
        try:
            windspeed_model_path = f"{self.args['models_save_path']}/{self.args['wind_speed_model_filename']}"
            input_size = 6
            hidden_size = 68
            num_layers = 2
            windspeed_model = WindSpeedPredictionLSTM(input_dim=input_size, hidden_dim=hidden_size, output_dim=1, num_layers=num_layers, dropout=0.2)
            windspeed_model.load_state_dict(torch.load(windspeed_model_path))
            windspeed_model.eval()
            windspeed_data = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)
            with torch.inference_mode():
                predicted_windspeed = windspeed_model(windspeed_data.unsqueeze(0))
            return predicted_windspeed.item()
        except Exception as e:
            raise CustomException(e, sys)
  
    def predict_temperature(self):
        try:
            model_path = f"{self.args['models_save_path']}/{self.args['temperature_model_filename']}"
            xgb_regressor = xgb.Booster()
            xgb_regressor.load_model(model_path)
            prediction_features  = self.data.to_numpy()
            prediction_features  = prediction_features[:, 2:].reshape(1,-1)

            prediction_features = xgb.DMatrix(prediction_features)
            predicted_temperature = xgb_regressor.predict(prediction_features)
            return predicted_temperature.item()

        except Exception as e:
            raise CustomException(e, sys)

    def predict_humidity(self):
        try:
            humidity_model_path = f"{self.args['models_save_path']}/{self.args['humidity_model_filename']}"
            input_dim = 24*7*5
            hidden_dim = 128
            output_dim = 1
            dropout = 0.2
            humidity_model = HumidityMLP(input_dim=input_dim,
                                         hidden_dim=hidden_dim,
                                         output_dim=output_dim,
                                         dropout=dropout)
            humidity_model.load_state_dict(torch.load(humidity_model_path))
            humidity_data = self.data.iloc[:, 2:].values.flatten()
            humidity_data  = torch.tensor(humidity_data, dtype=torch.float32)
            humidity_model.eval()
            with torch.inference_mode():
                predicted_humidity = humidity_model(humidity_data)
            capping_value = 100.0
            return min(predicted_humidity.item(), capping_value)
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_precipitation(self):
        try:
            model_path = f"{self.args['models_save_path']}/{self.args['precipitation_model_filename']}"
            randonforest_regressor = joblib.load(model_path)
            prediction_features  = self.data.to_numpy()
            prediction_features  = prediction_features[:, 4:].reshape(1, -1)
            predicted_precipitation = randonforest_regressor.predict(prediction_features)
            return predicted_precipitation.item()
        except Exception as e:
            raise CustomException(e, sys)
        

def prediction_pipeline():     
    object = PredictPipeline(args=args)
    object.data_prepration()
    wind = object.predict_wind_speed()
    temp = object.predict_temperature()
    humidity = object.predict_humidity()
    rain = object.predict_precipitation()
    return wind, temp, humidity, rain

