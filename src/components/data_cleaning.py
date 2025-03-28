import os
import sys

from pathlib import Path
import pandas as pd
import yaml


# Get the project root (weather-forcasting) and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import logging
from src.exception_handler import CustomException
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)


class Cleaner:
    def __init__(self, data_folder_path, processed_data_path, skip_rows, args = args, weather_parameters=None, prcp_colm=None):
        self.args = args
        self.data_folder_path = data_folder_path
        self.processed_data_path = processed_data_path
        self.weather_parameters = weather_parameters  
        self.time_col = self.weather_parameters[0]
        self.skip_rows = skip_rows
        self.prcp_colm = prcp_colm

    def initiate_import(self):
        logging.info("Raw dataset import for cleaning is initiated...")

        try:
            df_list = []
            count = 0

            for file in os.listdir(self.data_folder_path):
                if file.endswith("csv"):
                    file_path = os.path.join(self.data_folder_path, file)
                    try:
                        df = pd.read_csv(file_path, skiprows=self.skip_rows, na_values="NA", usecols=self.weather_parameters)
                    except:
                        df = pd.read_csv(file_path, skiprows=self.skip_rows, na_values="NA", usecols=self.prcp_colm)

                    df = df.drop(df.index[-1])  
                    df_list.append(df)
                    count += 1
                    logging.info(f"File {file} imported successfully")


            combined_df = pd.concat(df_list, ignore_index=True)
            try:
                combined_df[self.time_col] = pd.to_datetime(combined_df[self.time_col], format="%Y-%m-%d %H:%M:%S")
            except:
                combined_df[list(self.prcp_colm)[0]] = pd.to_datetime(combined_df[list(self.prcp_colm)[0]], format="%Y-%m-%d %H:%M:%S")

            
            logging.info(f"Successfully imported and concatenated {count} files")
            return combined_df

        except Exception as e:
            raise CustomException(e,sys)

    def fill_missing_values(self, df):
        logging.info("Data cleaning initiated: Handling missing values")

        try:
            if self.prcp_colm is None:
                pass
            else:
                self.time_col = list(self.prcp_colm)[0]

            df[self.time_col] = pd.to_datetime(df[self.time_col], format="%Y-%m-%d %H:%M:%S")

            for col in df.columns:
                for idx, row in df[df[col].isnull()].iterrows():
                    missing_time = row[self.time_col]
                    missing_hour = missing_time.hour
                    missing_day = missing_time.day
                    missing_month = missing_time.month
                    missing_year = missing_time.year

                    found_value = None

                    # Search within ±5 years
                    for offset in range(1, 6):
                        prev_year_data = df[
                            (df[self.time_col].dt.month == missing_month) &
                            (df[self.time_col].dt.day == missing_day) &
                            (df[self.time_col].dt.year == missing_year - offset) &
                            (df[self.time_col].dt.hour == missing_hour)
                        ]
                        if not prev_year_data.empty:
                            found_value = prev_year_data[col].values[0]
                            break

                        next_year_data = df[
                            (df[self.time_col].dt.month == missing_month) &
                            (df[self.time_col].dt.day == missing_day) &
                            (df[self.time_col].dt.year == missing_year + offset) &
                            (df[self.time_col].dt.hour == missing_hour)
                        ]
                        if not next_year_data.empty:
                            found_value = next_year_data[col].values[0]
                            break

                    if found_value is not None:
                        df.at[idx, col] = found_value

            missing_counts = df.isnull().sum().sum()
            if missing_counts > 0:
                logging.info(f"{missing_counts} missing values remain, applying forward fill...")
                df.ffill(inplace=True)

            df = df.drop_duplicates(subset=[self.time_col], keep='first').reset_index(drop=True)
            logging.info("Missing values and duplicates handled successfully")

            return df

        except Exception as e:
            raise CustomException(e,sys)

    def is_perfect_timeseries_and_fill(self, df):

        if self.prcp_colm is None:
            pass
        else:
            self.time_col = list(self.prcp_colm)[0]

        logging.info("Checking for gaps in time series and filling them...")
        try:
            # Ensure datetime type
            df[self.time_col] = pd.to_datetime(df[self.time_col])

            # Sort and reset index
            df = df.sort_values(by=self.time_col).reset_index(drop=True)

            # Find time differences
            df['time_diff'] = df[self.time_col].diff()

            # Expected frequency (assumed mode)
            mode_diff = df['time_diff'].value_counts().idxmax()

            # Locate gaps
            gaps = df[df['time_diff'] > mode_diff].index.tolist()

            if not gaps:
                logging.info("Perfect time series detected!")
                return df.drop(columns=['time_diff'])

            logging.warning(f"Inconsistent time intervals detected at: {gaps}")

            # Precompute monthly-hourly averages
            df['month'] = df[self.time_col].dt.month
            df['hour'] = df[self.time_col].dt.hour

            monthly_hourly_avg = df.groupby(['month', 'hour']).mean(numeric_only=True)

            # Fill gaps
            for idx in reversed(gaps):
                prev_time = df.at[idx - 1, self.time_col]
                current_time = df.at[idx, self.time_col]
                gap_hours = int((current_time - prev_time) / pd.Timedelta(hours=1)) - 1

                for hour in range(1, gap_hours + 1):
                    new_time = prev_time + pd.Timedelta(hours=hour)
                    filled_row = None

                    # 1. Try previous year data (T-365 days)
                    prev_year_time = new_time - pd.Timedelta(days=365)
                    prev_year_idx = df[df[self.time_col] == prev_year_time].index
                    if not prev_year_idx.empty:
                        filled_row = df.iloc[prev_year_idx[0]].copy()
                        filled_row[self.time_col] = new_time
                    else:
                        # 2. Fallback to monthly-hourly average
                        month, hour = new_time.month, new_time.hour
                        if (month, hour) in monthly_hourly_avg.index:
                            filled_row = monthly_hourly_avg.loc[(month, hour)].copy()
                            filled_row[self.time_col] = new_time
                        else:
                            logging.warning(f"No data found for {new_time}, even after fallbacks.")
                            continue

                    # Insert filled row
                    new_row = pd.DataFrame([filled_row])
                    df = pd.concat([df.iloc[:idx], new_row, df.iloc[idx:]]).reset_index(drop=True)
                    idx += 1  # Move forward for next insert

            # Drop helper columns
            df = df.drop(columns=['time_diff', 'month', 'hour'])

            # Final check
            if self.is_perfect_timeseries(df):
                logging.info("All gaps filled successfully!!")
            else:
                logging.warning("Some gaps could not be filled!!")
                
            file_save_path = Path(self.processed_data_path)
            file_save_path.mkdir(parents=True, exist_ok=True)

            if self.prcp_colm is None:
                df.to_csv(os.path.join(file_save_path,self.args["processed_weather_data_filename"]), index=False)
            else:
                df.to_csv(os.path.join(file_save_path,self.args["processed_rain_data_filename"]), index=False)

            logging.info(f"Cleaned file saved Successfully at location: {file_save_path}")
        
        except Exception as e:
            raise CustomException(e,sys)

    def is_perfect_timeseries(self, df):
        """
        Check whether a dataset is a perfect time series (fixed frequency).

        """

        try:
            if self.prcp_colm is None:
                pass
            else:
                self.time_col = list(self.prcp_colm)[0]

            duplicates = df[df[self.time_col].duplicated(keep='first')]
            if not duplicates.empty:
                logging.warning("Duplicate timestamps found!")
                return False

            df = df.sort_values(by=self.time_col).reset_index(drop=True)
            inferred_freq = pd.infer_freq(df[self.time_col])

            if inferred_freq is None:
                logging.warning("Inconsistent time intervals detected.")
                return False

            return True
        except Exception as e:
            raise CustomException(e,sys)
        
    def merge(self):
        logging.info("Initiate merging of rain data and other weather parameters")
        try:
            processed_rain_data = pd.read_csv(os.path.join(self.args["processed_data_save_path"],
                                                           self.args["processed_rain_data_filename"]))
            processed_rain_data[self.args["rain_parameters"][0]] = pd.to_datetime(processed_rain_data[self.args["rain_parameters"][0]],
                                                                             format = "%Y-%m-%d %H:%M:%S")
            logging.info(f"Processed rain data imported successfully")

            processed_weather_data = pd.read_csv(os.path.join(self.args["processed_data_save_path"],
                                                              self.args["processed_weather_data_filename"]))
            processed_weather_data[self.args["weather_parameters"][0]] = pd.to_datetime(processed_weather_data[self.args["weather_parameters"][0]],
                                                                                   format="%Y-%m-%d %H:%M:%S")
            logging.info(f"Processed weather data imported successfully")

            merged_df = processed_weather_data.merge(processed_rain_data[self.args["rain_parameters"]],
                                                     left_on=self.args["weather_parameters"][0],
                                                     right_on=self.args["rain_parameters"][0],
                                                     how="left")
            

            merged_df.drop(columns=[self.args["rain_parameters"][0]], inplace=True)
            return merged_df
        

        except Exception as e:
            raise CustomException(e, sys) 
    


def pipeline():

    weather_data_cleaner = Cleaner(
        data_folder_path=args["weather_data_save_path"],
        processed_data_path=args["processed_data_save_path"],
        weather_parameters=args["weather_parameters"],
        skip_rows=args["weather_data_skiprows"]
    )
    weather_data = weather_data_cleaner.initiate_import()
    weather_data = weather_data_cleaner.fill_missing_values(df=weather_data)
    weather_data_cleaner.is_perfect_timeseries_and_fill(weather_data)


    rain_data_cleaner = Cleaner(
        data_folder_path=args["rain_data_save_path"],
        processed_data_path=args["processed_data_save_path"],
        skip_rows=args["rain_data_skiprows"],
        weather_parameters=args["weather_parameters"],
        prcp_colm=args["rain_parameters"]
    )

    rain_data = rain_data_cleaner.initiate_import()
    rain_data = rain_data_cleaner.fill_missing_values(df=rain_data)
    rain_data_cleaner.is_perfect_timeseries_and_fill(df=rain_data)
    rain_data = rain_data_cleaner.merge()

        



