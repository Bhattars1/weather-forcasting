import os
import yaml
import sys
from datetime import timedelta

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import logging
from src.exception_handler import CustomException
from src.pipeline.predict_pipeline import prediction_pipeline


# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

class MultiHourForcast:
    def __init__(self,
                 args=args):
        self.args = args
    
    def forcasting(self):
        try:

            for i in tqdm(range(self.args["forcasting_hours"])):

                logging.info("Forcasting for the next hour")
                wind, temp, humidity, rain = prediction_pipeline()

                logging.info("Forcasting for the next hour completed")
                logging.info(f"The predicted values are wind: {wind}, temperature: {temp}, humidity: {humidity}, precipation: {rain}")

                # Importing the file to update
                logging.info("Importing the file to update the predicted values")
                forcasting_df = pd.read_csv(args["forcasting_data_path"])

                columns = forcasting_df.columns.tolist()
                forcasting_df[columns[0]] = pd.to_datetime(forcasting_df[columns[0]], dayfirst=True)
                
                recent_time = forcasting_df[columns[0]].max()
                recent_time = pd.to_datetime(recent_time, dayfirst=True)

                recent_time = recent_time + timedelta(hours=1)
                forcasting_df.loc[len(forcasting_df)] = [recent_time, wind, temp, humidity, rain]


                logging.info("Added the predicted values to the dataframe")

                forcasting_df.drop(0, axis=0, inplace=True)
                forcasting_df.to_csv(args["forcasting_data_path"], index=False)
                logging.info(f"Saved the file with the predicted values of the hour {i+1}")
            
            logging.info("Forcasting completed!!!!!!!!!!!!!!!!!")
                
        except Exception as e:
            raise CustomException(e, sys)
        
    def plot_forcasted_data(self):
        try:
            df = pd.read_csv(args["forcasting_data_path"])
            df_forcasted = df.tail(24)
            df_forcasted.iloc[:,0] = pd.to_datetime(df_forcasted.iloc[:,0], format="%Y-%m-%d %H:%M:%S")

            for col in df.columns:
                
                if col == "ob_time":
                    continue
                plt.figure(figsize=(16, 10))
                plt.plot(df_forcasted["ob_time"], df_forcasted[col])
                plt.xlabel("Time")
                plt.ylabel(col)
                plt.title(f"{col} forcasting")
                plt.grid()
                plt.show()


            logging.info("Plotting completed!!!!!!!!!!!!!!!!!")

        except Exception as e:
            raise CustomException(e, sys)


        
forcasting = MultiHourForcast(args)
forcasting.forcasting()
forcasting.plot_forcasted_data()
    
            
