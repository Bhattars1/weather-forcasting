import sys
import os
import yaml

import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import logging
from src.exception_handler import CustomException

with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

scaler = joblib.load(args["scalar_path"])


class PredictPreprocessing:
    def __init__(self,
                 data_path=args["forcasting_data_path"],
                 scaler=scaler,
                 args = args,
                 data = None
                 ):
        self.data_path = data_path
        self.data = data
        self.scaler = scaler
        self.args = args

    def transformations(self):
        try:
            logging.info("Transformations of day and month initiated")
            self.data = pd.read_csv(self.data_path)

            
            # Rows Consistency
            expected_rows = 7*24
            if len(self.data) > expected_rows:
                logging.info(f"Input limit exceeded!!! Truncating data from {len(self.data)} to last {expected_rows} rows.")
                self.data = self.data.tail(expected_rows).reset_index(drop=True)

        
            datetime_col = self.data.columns[0]
            self.data[datetime_col] = pd.to_datetime(self.data[datetime_col], format="%Y-%m-%d %H:%M:%S")

            # Extract hour and month
            self.data["hour"] = self.data.iloc[:, 0].dt.hour
            self.data["month"] = self.data.iloc[:, 0].dt.month


            # Fourier Transform Encoding for Hour (24-hour cycle)
            self.data["sine_hour"] = np.sin(2 * np.pi * self.data["hour"] / 24)

            # Fourier Transform Encoding for Month (12-month cycle)
            self.data["sine_month"] = np.sin(2 * np.pi * self.data["month"] / 12)

            self.data.to_csv("seed.csv", index=False)
            
            logging.info("Successfully transformed hour and month for prediction")

            time_col = self.data["ob_time"]
            self.data.drop(columns=["hour", "month", "ob_time"], inplace=True)
            # Scale the data

            logging.info("Scaling the data")
            self.data = self.data.to_numpy()
            # np.savetxt("seed.csv", self.data, delimiter=",")

            self.data = self.scaler.transform(self.data)

            self.data = pd.DataFrame(self.data)
            self.data.insert(0, "ob_time", time_col)  

            logging.info("Successfully scaled the data")

            return self.data
        except Exception as e:
            raise CustomException(e, sys)
             
    # def prediction_data_consistency(self):
        logging.info("Checking data consistency for prediction")

        try:
            # Rows Consistency
            expected_rows = 7*24
            if len(self.data) > expected_rows:
                logging.info(f"Input limit exceeded!!! Truncating data from {len(self.data)} to last {expected_rows} rows.")
                self.data = self.data.tail(expected_rows).reset_index(drop=True)

        except Exception as e:
            raise CustomException(e, sys)
        
        
        try:
            float_cols = self.data.columns[1:]
            for col in float_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            self.data[datetime_col] = pd.to_datetime(self.data[datetime_col], errors='coerce', dayfirst=True)

            # Columns consistency
            expeced_columns = self.args["weather_parameters"].copy()
            expeced_columns.append(self.args["rain_parameters"][1])
            missing_cols = [col for col in expeced_columns if col not in self.data.columns]
            
            if "ob_time" in missing_cols:
                logging.warning(f"Date and time is missing. Forcasting is not possible!!!")
                raise Exception("Date and time is missing. Forcasting is not possible!!!")
            if len(missing_cols) > 0:
                logging.info(f"Missing parameter/s: {missing_cols} found!! This may lead to less accurate forcast...")

            # Fill up precipation feature if missing
            if "prcp_amt" in missing_cols:
                data_search_path = self.args["rain_data_save_path"]
                df_list = []
                for file in os.listdir(data_search_path):
                    if file.endswith("csv"):
                        file_path = os.path.join(data_search_path, file)

                        df = pd.read_csv(file_path, skiprows=self.args["rain_data_skiprows"], na_values="NA", usecols=self.args["rain_parameters"])
                        df = df.drop(df.index[-1])  
                        df_list.append(df)

                combined_df = pd.concat(df_list, ignore_index=True)
                datetime_col = self.args["rain_parameters"][0]
                rain_col = self.args["rain_parameters"][1]
                combined_df[datetime_col] = pd.to_datetime(combined_df[datetime_col], errors="coerce")
                combined_df.sort_values(by=datetime_col, inplace=True)


                # Step 1: Get the date range to match from main data
                main_datetime = pd.to_datetime(self.data.iloc[:, 0], errors="coerce")
                target_dates = main_datetime.dt.strftime("%m-%d %H:%M:%S").unique()  # Only MM-DD HH:MM:SS matters

                # Step 2: Extract year range to fallback
                years = combined_df[datetime_col].dt.year.dropna().unique()
                years = sorted(years, reverse=True)  # Start from most recent year

               
                for y in years:
                    # Filter for that year and the date + time pattern
                    subset = combined_df[combined_df[datetime_col].dt.year == y]
                    subset = subset[subset[datetime_col].dt.strftime("%m-%d %H:%M:%S").isin(target_dates)]

                    if len(subset) >= 168:  # Check for complete week (7 days * 24 hours)
                        # Attempt to fill NaN from previous years
                        for date in target_dates:
                            if date in subset[datetime_col].dt.strftime("%m-%d %H:%M:%S").values:
                                index = subset[datetime_col].dt.strftime("%m-%d %H:%M:%S").values.tolist().index(date)
                                if pd.isna(subset[rain_col].iloc[index]):
                                    # Look at previous year data for that exact date
                                    for prev_year in years:
                                        if prev_year == y:
                                            continue
                                        prev_year_subset = combined_df[combined_df[datetime_col].dt.year == prev_year]
                                        prev_year_subset = prev_year_subset[
                                            prev_year_subset[datetime_col].dt.strftime("%m-%d %H:%M:%S") == date]
                                        if len(prev_year_subset) > 0:
                                            # If previous year data exists, fill it
                                            if not pd.isna(prev_year_subset[rain_col].iloc[0]):
                                                subset[rain_col].iloc[index] = prev_year_subset[rain_col].iloc[0]
                                                break

                        # Now update self.data with the found data
                        self.data["prcp_amt"] = subset[rain_col].reset_index(drop=True)

                logging.info("Successfully filled the precipation input data")


            
            if "prcp_amt" in missing_cols:
                missing_cols.remove("prcp_amt")

            if "prcp_amt" in self.args["weather_parameters"]:
                self.args["weather_parameters"].remove("prcp_amt")
                
            # fill up other weather features if missing(excluding prcp_amt)
            data_search_path = self.args["weather_data_save_path"]
            df_list = []

            for file in os.listdir(data_search_path):
                if file.endswith("csv"):
                    file_path = os.path.join(data_search_path, file)
                    df = pd.read_csv(
                        file_path,
                        skiprows=self.args["weather_data_skiprows"],
                        na_values="NA",
                        usecols=self.args["weather_parameters"] # Load all weather parameters
                    )
                    df = df.drop(df.index[-1])  # Remove last row (summary/empty)
                    df_list.append(df)

                # Combine all the weather data into one dataframe
                combined_df = pd.concat(df_list, ignore_index=True)
                datetime_col = "ob_time"
                combined_df[datetime_col] = pd.to_datetime(combined_df[datetime_col], errors="coerce")
                combined_df.sort_values(by=datetime_col, inplace=True)

                # Step 1: Get the date range to match from main data
                main_datetime = pd.to_datetime(self.data.iloc[:, 0], errors="coerce")
                target_dates = main_datetime.dt.strftime("%m-%d %H:%M:%S").unique()  # Only MM-DD HH:MM:SS matters

                # Step 2: Extract year range to fallback
                years = combined_df[datetime_col].dt.year.dropna().unique()
                years = sorted(years, reverse=True) 
                
            for param in self.args["weather_parameters"]:  # Loop through all weather parameters
                if param in missing_cols:
                    logging.info(f"Missing parameter: {param} found!")
                    
                    for y in years:
                        # Filter for that year and the date + time pattern
                        subset = combined_df[combined_df[datetime_col].dt.year == y]
                        subset = subset[subset[datetime_col].dt.strftime("%m-%d %H:%M:%S").isin(target_dates)]

                        if len(subset) >= 168:  # Check for complete week (7 days * 24 hours)
                            # Attempt to fill NaN from previous years
                            for date in target_dates:
                                if date in subset[datetime_col].dt.strftime("%m-%d %H:%M:%S").values:
                                    index = subset[datetime_col].dt.strftime("%m-%d %H:%M:%S").values.tolist().index(date)
                                    if pd.isna(subset[param].iloc[index]):
                                        # Look at previous year data for that exact date
                                        for prev_year in years:
                                            if prev_year == y:
                                                continue
                                            prev_year_subset = combined_df[combined_df[datetime_col].dt.year == prev_year]
                                            prev_year_subset = prev_year_subset[
                                                prev_year_subset[datetime_col].dt.strftime("%m-%d %H:%M:%S") == date]
                                            if len(prev_year_subset) > 0:
                                                # If previous year data exists, fill it
                                                if not pd.isna(prev_year_subset[param].iloc[0]):
                                                    subset[param].iloc[index] = prev_year_subset[param].iloc[0]
                                                    break

                            # Now update self.data with the found data for this weather parameter
                            self.data[param] = subset[param].reset_index(drop=True)
                            break 
                    logging.info(f"Successfully filled the {param} input data")

        except Exception as e:
            raise CustomException(e,sys)

        logging.info("Data null value checking started")
        try:

            # Check if any of the data are missing
            if self.data.isnull().values.any():
                logging.info("Null value found, Starting  process.")
                self.data.ffill()
            else:
                logging.info("No null value found proceeding further...")

        except Exception as e:
            raise CustomException(e, sys)
    
        

def prediction_preprocessing_pipeline():

    obj = PredictPreprocessing()
    # obj.prediction_data_consistency()
    data = obj.transformations()
    return data

    
