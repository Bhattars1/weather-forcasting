import os
import yaml
import sys
from datetime import timedelta

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pipeline.data_ingestion_pipeline import GetWeather
from src.logger import logging
from src.exception_handler import CustomException
from src.pipeline.predict_pipeline import prediction_pipeline


# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

get_weather = GetWeather(save_path= args["ingested_save_path"])
get_test_weather = GetWeather(save_path= args["ingested_test_save_path"], start_day=2, end_day=2)

class MultiHourForcast:
    def __init__(self,
                 args=args):
        self.args = args
    
    def forcasting(self):
        try:
            logging.info("Forcasting process initiated")
            # Get the weather data
            get_weather.get_weather()
            get_test_weather.get_weather()

            for i in tqdm(range(self.args["forcasting_hours"])):

                logging.info("Forcasting for the next hour")
                wind, temp, humidity, rain = prediction_pipeline()

                logging.info("Forcasting for the next hour completed")
                logging.info(f"The predicted values are wind: {wind}, temperature: {temp}, humidity: {humidity}, precipation: {rain}")

                # Importing the file to update
                logging.info("Importing the file to update the predicted values")
                forcasting_df = pd.read_csv(args["forcasting_data_path"])

                columns = forcasting_df.columns.tolist()
                forcasting_df[columns[0]] = pd.to_datetime(forcasting_df[columns[0]], format = "%Y-%m-%d %H:%M:%S")
                
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
        
    def plot_forecasted_vs_actual(self):
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.read_csv(self.args["forcasting_data_path"])
            test_df = pd.read_csv("data/forcasting_data/test_data.csv")

            df = df.tail(24)  # last 24 hours of prediction
            df["ob_time"] = pd.to_datetime(df["ob_time"], format="%Y-%m-%d %H:%M:%S")
            test_df["ob_time"] = pd.to_datetime(test_df["ob_time"], format="%Y-%m-%d %H:%M:%S")

            # Merge on ob_time
            merged_df = pd.merge(df, test_df, on="ob_time", suffixes=("_pred", "_actual"))


            variables = [col for col in df.columns if col != "ob_time"]

            fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(16, 4 * len(variables)), sharex=True)

            if len(variables) == 1:
                axes = [axes]

            for ax, var in zip(axes, variables):
                ax.plot(merged_df["ob_time"], merged_df[f"{var}_actual"], label="Actual", color="blue")
                ax.plot(merged_df["ob_time"], merged_df[f"{var}_pred"], label="Predicted", linestyle='--', color="orange")
                ax.set_ylabel(var)
                ax.set_title(f"{var} - Actual vs Predicted")
                ax.grid(True)
                ax.legend()

            axes[-1].set_xlabel("Time")

            plt.tight_layout()
            plt.show()

            save_option = input("Do you want to save the plot? (y/n): ").strip().lower()
            if save_option == "y":
                file_name = self.args["forcasted_plot_path"]
                fig.savefig(f"{file_name}.png")
                logging.info(f"Plot saved as {file_name}.png")
            else:
                logging.info("Plot not saved.")

            logging.info("Plotting completed.")

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    forcast = MultiHourForcast()
    forcast.forcasting()
    forcast.plot_forecasted_vs_actual()
            
