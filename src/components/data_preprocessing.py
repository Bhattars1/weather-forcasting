import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

from src.logger import logging
from src.exception_handler import CustomException
from src.components.data_cleaning import cleaning_pipeline

# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

scalar = MinMaxScaler()
class Preprocessing:

    def __init__(self, cleaning_function=cleaning_pipeline, scaler=scalar ,args=args):
        self.args = args
        self.cleaning_function = cleaning_function
        self.scaler = scaler
    
    def trigonometric_transfom(self):
        try:  
            df = self.cleaning_function()
            df[args["weather_parameters"][0]]=pd.to_datetime(df[args["weather_parameters"][0]], format = "%Y-%m-%d %H:%M:%S")
            
            # Extract hour and month
            df["hour"] = df["ob_time"].dt.hour
            df["month"] = df["ob_time"].dt.month

            # Fourier Transform Encoding for Hour (24-hour cycle)
            df["sine_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
            # df["cosine_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
            logging.info("Successfully performed Fourier transform of hour of the day")

            # Fourier Transform Encoding for Month (12-month cycle)
            df["sine_month"] = np.sin(2 * np.pi * df["month"] / 12)
            # df["cosine_month"] = np.cos(2 * np.pi * df["month"] / 12)
            logging.info("Successfully performed Fourier transform of month of the year")


            try:
                # Cyclic Encoding for Wind Direction
                df["wind_dir_sin"] = np.sin(np.radians(df[args["weather_parameters"][1]]))
                logging.info("successfully encoded the wind direction using sine function")

                return df
            except:
                print("Could not transform wind direction to cyclic encoding, Continue to next step...")
                pass
            # Drop original columns that are now encoded
            df.drop(columns=["ob_time", "hour", "month"], inplace=True)

        except Exception as e:
            raise CustomException(e, sys) 

    def sliding_window_dataset(self):
        try:
            df = self.trigonometric_transfom()
            logging.info("Sliding window preprocessing step initiated...") 

            # Define constants
            data_input = 7*24
            data_output = 1


            # Compute total time window size
            time_window = (data_input + data_output)
            num_samples = len(df) - time_window

            # Preallocate arrays for efficiency
            X = np.zeros((num_samples, data_input, df[args["features"]].shape[1]))
            Y = np.zeros((num_samples, data_output , df[args["targets"]].shape[1]))
            
            # Loop through the dataset to extract sequences
            for i in range(num_samples):
                X[i] = df[args["features"]][i : i + data_input ]  
                Y[i] = df[args["targets"]][i + data_input : i + time_window] 

            logging.info("Sliding window method successfully executed")
            logging.info(f"Shape of data: {X.shape}")
            logging.info(f"Shape of labels: {Y.shape}")
            
            X_reshaped = X.reshape(-1, X.shape[2])
            X_scaled = self.scaler.fit_transform(X=X_reshaped)

            # Reshape back to the original shape
            X = X_scaled.reshape(X.shape)

            logging.info("The Features are scaled Successfully")

            # Save the preprocessed dataset
            directory = './data/preprocessed_data'
            if not os.path.exists(directory):
                os.makedirs(directory)
            logging.info(f"Directory '{directory}' created.")
            with open('./data/preprocessed_data/data.pkl', 'wb') as f:
                pickle.dump(X, f)
            with open('./data/preprocessed_data/labels.pkl', 'wb') as f:
                pickle.dump(Y, f)
            logging.info("The preprocessed data and labels are saved successfully in the directory")
        
        except Exception as e:
            raise CustomException(e,sys)
        

def preprocessing_pipeline():
    obj = Preprocessing()
    obj.sliding_window_dataset()
    