import os
import sys
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import logging
from src.exception_handler import CustomException
from src.components.data_cleaning import cleaning_pipeline

# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)


class Preprocessing:

    def __init__(self, cleaning_function=cleaning_pipeline, args=args):
        self.args = args
        self.cleaning_function = cleaning_function()

    def load_data(self):
        return cleaning_pipeline()
    
    def trigonometric_transfom(self):
        try:  
            df = self.load_data()
            df[args["weather_parameters"][0]]=pd.to_datetime(df[args["weather_parameters"][0]], format = "%Y-%m-%d %H:%M:%S")
            
            # Extract hour and month
            df["hour"] = df["ob_time"].dt.hour
            df["month"] = df["ob_time"].dt.month

            # Fourier Transform Encoding for Hour (24-hour cycle)
            df["sine_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["cosine_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
            logging.info("Successfully performed Fourier transform of hour of the day")

            # Fourier Transform Encoding for Month (12-month cycle)
            df["sine_month"] = np.sin(2 * np.pi * df["month"] / 12)
            df["cosine_month"] = np.cos(2 * np.pi * df["month"] / 12)
            logging.info("Successfully performed Fourier transform of month of the year")


            # Cyclic Encoding for Wind Direction
            df["wind_dir_sin"] = np.sin(np.radians(df[args["weather_parameters"][1]]))
            logging.info("successfully encoded the wind direction using sine function")

            # Drop original columns that are now encoded
            df.drop(columns=["ob_time", "hour", "month"], inplace=True)

            return df
        
        except Exception as e:
            raise CustomException(e, sys) 
        
    def sliding_window_dataset(self):
        try:
            df = self.trigonometric_transfom()
            logging.info("Sliding window preprocessing step initiated...") 

            # Define constants
            days_input = 10
            days_output = 1
            hours_per_day = 24

            # Compute total time window size
            time_window = (days_input + days_output) * hours_per_day  # 11 days of data
            num_samples = len(df) - time_window

            # Preallocate arrays for efficiency
            X = np.zeros((num_samples, days_input * hours_per_day, df[args["features"]].shape[1]))
            Y = np.zeros((num_samples, days_output * hours_per_day, df[args["targets"]].shape[1]))
            
            # Loop through the dataset to extract sequences
            for i in range(num_samples):
                X[i] = df[args["features"]][i : i + days_input * hours_per_day]  # 10 days of data
                Y[i] = df[args["targets"]][i + days_input * hours_per_day : i + time_window]  # 11th day

            logging.info("Sliding window method successfully executed")
            logging.info(f"Shape of data: {X.shape}")
            logging.info(f"Shape of labels: {Y.shape}")

            return X, Y
        
        except Exception as e:
            raise CustomException(e,sys)