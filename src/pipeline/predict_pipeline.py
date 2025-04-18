import os
import yaml
import sys

import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.components.algorithms import WindSpeedPredictionLSTM, HumidityMLP
from src.logger import logging
from src.exception_handler import CustomException
from src.components.algorithms import PrecipationPredictionLSTM, TemperaturePredictionLSTM
from src.components.data_preprocessing import Preprocessing
from src.pipeline.prediction_preprocessing import PredictPreprocessing
# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)



class PredictPipeline:
    def __init__(self, humidity_model,
                 precipation_mode,
                 wind_speed_model,
                 temperature_model,
                 args,
                 input_data
                 ):
        self.humidity_model = humidity_model
        self.precipation_model = precipation_mode
        self.wind_speed_model = wind_speed_model
        self.temperature_model = temperature_model
        self.args = args
        self.input_data = input_data
    
    

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

            
            pass
        except:
            pass
