import sys
import os
import yaml

import pandas as pd
import numpy as np
import joblib
from datetime import timedelta



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import logging
from src.exception_handler import CustomException

with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

scaler = joblib.load(args["scalar_path"])


data = pd.read_csv(r"src\pipeline\sample_data.csv")

class PredictPreprocessing:
    def __init__(self,
                 data : pd.DataFrame,
                 scaler=scaler,
                 args = args
                 ):
        self.data = data
        self.scaler = scaler
        self.args = args

    def trignometric_transforms(self):
        try:
            logging.info("Preprocessing for prediction initiated")

            # Extract hour and month
            self.data["hour"] = self.data.iloc[:, 0].dt.hour
            self.data["month"] = self.data.iloc[:, 0].dt.month

            # Fourier Transform Encoding for Hour (24-hour cycle)
            self.data["sine_hour"] = np.sin(2 * np.pi * self.data["hour"] / 24)

            # Fourier Transform Encoding for Month (12-month cycle)
            self.data["sine_month"] = np.sin(2 * np.pi * self.data["month"] / 12)
            logging.info("Successfully transformed hour and month for prediction")

            try:
                # Cyclic Encoding for Wind Direction
                self.data.iloc[:,1] = np.sin(np.radians(self.data.iloc[:,1]))
                logging.info("successfully encoded the wind direction using sine function")

                # Drop original columns that are now encoded
                self.data.drop(columns=["hour", "month"], inplace=True)
                self.data.drop(self.data.columns[0], axis=1, inplace=True)

            except Exception as e:
                raise CustomException(e, sys)
            
            try:
                self.data = self.data.to_numpy()
                scaled_data = self.scaler.transform(X=self.data)
                self.data = scaled_data.reshape(self.data.shape)
                logging.info("Scaled the data using MinMaxScaler for prediction")
                np.savetxt("output.csv", self.data, delimiter=",", fmt="%.9f")

            except Exception as e:
                raise CustomException(e, sys)
            logging.info("Preprocessing for prediction completed!!!")
            return self.data
            
        except Exception as e:
            raise CustomException(e, sys)
        

        
    def prediction_data_consistency(self):
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
            self.data[float_cols] = self.data[float_cols].astype(np.float64)
            datetime_col = self.data.columns[0]
            self.data[datetime_col] = pd.to_datetime(self.data[datetime_col], errors='coerce', dayfirst=True)

            # Columns consistency
            expeced_columns = self.args["weather_parameters"]
            expeced_columns.append(self.args["rain_parameters"][1])
            missing_cols = [col for col in expeced_columns if col not in self.data.columns]
            
            if "ob_time" in missing_cols[0]:
                logging.info(f"Date and time is missing. Forcasting is not possible!!!")
                raise CustomException("Time and Date is missing. Forcasting is not possible!!!", sys)

            logging.info(f"Missing parameter/s: {missing_cols} found!! This may lead to less accurate forcast...")

            try:
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

                    found = False
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
                            found = True
                            break
                    logging.info("Successfully filled the precipation input data")

            except Exception as e:
                raise CustomException(e, sys)
            
            if "prcp_amt" in missing_cols:
                missing_cols.remove("prcp_amt")

            if "prcp_amt" in self.args["weather_parameters"]:
                self.args["weather_parameters"].remove("prcp_amt")
                
            
            try:
                # Load the data only once
                data_search_path = self.args["weather_data_save_path"]
                df_list = []

                for file in os.listdir(data_search_path):
                    if file.endswith("csv"):
                        file_path = os.path.join(data_search_path, file)
                        df = pd.read_csv(
                            file_path,
                            skiprows=self.args["weather_data_skiprows"],
                            na_values="NA",
                            usecols=self.args["weather_parameters"][:-1] # Load all weather parameters
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
                    years = sorted(years, reverse=True)  # Start from most recent year

                for param in self.args["weather_parameters"]:  # Loop through all weather parameters
                    if param in missing_cols:
                        logging.info(f"Missing parameter: {param} found!")
                        found = False
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
                                found = True
                                break 
                        logging.info(f"Successfully filled the {param} input data")
            except Exception as e:
                raise CustomException(e, sys)
            self.data.to_csv("output.csv", index=False, header=True, float_format="%.9f")

        except Exception as e:
            raise CustomException(e,sys)


        try:
            logging.info("Starting missing weather & precipitation filling process.")

            # === 1. Parse datetime ===
            datetime_col = self.data.columns[0]
            self.data[datetime_col] = pd.to_datetime(self.data[datetime_col], errors='coerce', dayfirst=True)
            self.data.dropna(subset=[datetime_col], inplace=True)
            self.data.sort_values(by=datetime_col, inplace=True)

            # === 2. Data type conversion ===
            weather_cols = self.args["weather_parameters"][1:]  # skip ob_time
            rain_cols = self.args["rain_parameters"][1:]
            all_float_cols = self.data.columns[1:]  
            self.data[all_float_cols] = self.data[all_float_cols].astype(np.float64)

            logging.info(f"Date range in dataset: {self.data[datetime_col].min()} to {self.data[datetime_col].max()}")

            # === 3. Check for missing rows (expecting 168 for 7 days hourly) ===
            expected_len = 128
            actual_len = len(self.data)
            if actual_len < expected_len:
                logging.warning(f"Missing rows detected. Expected 168, got {actual_len}.")
                missing_start = self.data[datetime_col].min()
                missing_end = self.data[datetime_col].max()
                logging.warning(f"Incomplete date range: {missing_start} to {missing_end}")

            # === 4. Check for missing values ===
            missing_info = self.data.columns.isna().sum()
            missing_cols = missing_info[missing_info > 0]
            if not missing_cols.empty:
                logging.warning("Missing values found in the following columns:")
                for col, count in missing_cols.items():
                    logging.warning(f"   - {col}: {count} missing")

            # === 5. Load historical weather data ===
            past_weather = []
            for file in os.listdir(self.args["weather_data_save_path"]):
                if file.endswith(".csv"):
                    df = pd.read_csv(
                        os.path.join(self.args["weather_data_save_path"], file),
                        skiprows=self.args["weather_data_skiprows"],
                        na_values="NA",
                        usecols=self.args["weather_parameters"]
                    )
                    df = df.drop(df.index[-1])
                    past_weather.append(df)
            past_weather = pd.concat(past_weather, ignore_index=True)
            past_weather["ob_time"] = pd.to_datetime(past_weather["ob_time"], errors='coerce', dayfirst=True)
            past_weather.dropna(subset=["ob_time"], inplace=True)
            past_weather.sort_values("ob_time", inplace=True)

            # === 6. Load historical rain data ===
            past_rain = []
            for file in os.listdir(self.args["rain_data_save_path"]):
                if file.endswith(".csv"):
                    df = pd.read_csv(
                        os.path.join(self.args["rain_data_save_path"], file),
                        skiprows=self.args["rain_data_skiprows"],
                        na_values="NA",
                        usecols=self.args["rain_parameters"]
                    )
                    df = df.drop(df.index[-1])
                    past_rain.append(df)
            past_rain = pd.concat(past_rain, ignore_index=True)
            past_rain["ob_time"] = pd.to_datetime(past_rain["ob_time"], errors='coerce', dayfirst=True)
            past_rain.dropna(subset=["ob_time"], inplace=True)
            past_rain.sort_values("ob_time", inplace=True)

            # === 7. Create key column ===
            self.data["key"] = self.data[datetime_col].dt.strftime("%m-%d %H:%M")
            past_weather["key"] = past_weather["ob_time"].dt.strftime("%m-%d %H:%M")
            past_rain["key"] = past_rain["ob_time"].dt.strftime("%m-%d %H:%M")

            # === 8. Fill missing rows (reindex to full hourly range) ===
            full_range = pd.date_range(self.data[datetime_col].min(), self.data[datetime_col].max(), freq='H')
            self.data = self.data.set_index(datetime_col).reindex(full_range).reset_index()
            self.data.rename(columns={'index': datetime_col}, inplace=True)
            self.data["key"] = self.data[datetime_col].dt.strftime("%m-%d %H:%M")

            # === 9. Fill missing weather columns ===
            for col in weather_cols:
                self.data[col] = self.data.apply(
                    lambda row: past_weather.loc[past_weather["key"] == row["key"], col].dropna().iloc[0]
                    if pd.isna(row[col]) and col in past_weather.columns and not past_weather.loc[past_weather["key"] == row["key"], col].dropna().empty
                    else row[col],
                    axis=1
                )

            # === 10. Fill missing rain columns ===
            for col in rain_cols:
                self.data[col] = self.data.apply(
                    lambda row: past_rain.loc[past_rain["key"] == row["key"], col].dropna().iloc[0]
                    if pd.isna(row[col]) and col in past_rain.columns and not past_rain.loc[past_rain["key"] == row["key"], col].dropna().empty
                    else row[col],
                    axis=1
                )

            # === 11. Drop the key column ===
            self.data.drop(columns=["key"], inplace=True)

            # === 12. Save cleaned data ===
            self.data.to_csv("cleaned_weather_and_precip.csv", index=False)
            logging.info("Filling complete. Cleaned data saved to cleaned_weather_and_precip.csv")

        except Exception as e:
            raise CustomException(e, sys)
        
            

obj = PredictPreprocessing(data=data)
obj.prediction_data_consistency()
obj.trignometric_transforms()