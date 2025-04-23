import pandas as pd
import yaml
import sys
import os
from datetime import datetime, timedelta, UTC


import openmeteo_requests
import requests_cache
from retry_requests import retry

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.exception_handler import CustomException
from src.logger import logging

with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)



wind_speed = args["weather_parameters"][1]
temperature = args["weather_parameters"][2]
relative_humidity = args["weather_parameters"][3]
precipitation = args["rain_parameters"][1]

columns = [wind_speed, temperature, relative_humidity, precipitation]

def get_forecast_date_range():
    today = datetime.now(UTC).date()   
    start_date = today - timedelta(days=9)     
    end_date = today - timedelta(days=2)  
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

start_date, end_date = get_forecast_date_range()
    
ingestion_params = {
    "latitude": 51.4787,
    "longitude": -0.2956,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
}

class GetWeather:
    def __init__(self, ingestion_params=ingestion_params, columns=columns):
        self.ingestion_params = ingestion_params
        self.columns = columns

    def get_weather(self):
        try:
            logging.info("Weather data retriving process initiated")
            # Setup the Open-Meteo API client with cache and retry on error
            cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
            retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
            openmeteo = openmeteo_requests.Client(session = retry_session)

            responses = openmeteo.weather_api(url="https://archive-api.open-meteo.com/v1/archive", params=self.ingestion_params)

            # Process first location. Add a for-loop for multiple locations or weather models
            response = responses[0]
            
            logging.info(f"Coordinates {response.Latitude()}°N {response.Longitude()}°W")
            logging.info(f"Elevation {response.Elevation()} m asl")

            # Process hourly data. The order of variables needs to be the same as requested.
            hourly = response.Hourly()
            hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
            hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
            hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
            hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
            logging.info(f"Weather data retrieved successfully")

            hourly_data = {"ob_time": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = False),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = False),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )}

            hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
            hourly_data["temperature_2m"] = hourly_temperature_2m
            hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
            hourly_data["precipitation"] = hourly_precipitation

            hourly_dataframe = pd.DataFrame(data = hourly_data)
            hourly_dataframe.rename(columns={"wind_speed_10m": self.columns[0],
                                            "temperature_2m": self.columns[1],
                                            "relative_humidity_2m": self.columns[2],
                                            "precipitation": self.columns[3],
                                            }, inplace=True)
            save_path = "data/forcasting_data/weekly_data.csv"
            hourly_dataframe.to_csv(save_path, index = False)

            logging.info(f"Weather data saved successfully to {save_path}")
        except Exception as e:
            raise CustomException(e, sys)