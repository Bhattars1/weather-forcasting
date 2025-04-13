import os
import sys

from bs4 import BeautifulSoup
import requests
from pathlib import Path

# Get the project root (weather-forcasting) and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import logging
from src.exception_handler import CustomException


class DataIngestion:

    def __init__(self,
                 save_path,
                 payload,
                 data_base_url,
                 login_url,
                 year_range,
                 headers,
                 ):
        self.payload = payload
        self.data_base_url = data_base_url
        self.login_url = login_url
        self.year_range = year_range
        self.headers = headers
        self.save_path = save_path

    def initiate_scrapping(self):
        logging.info("Data ingestion initiated")
        try:
            session = requests.Session()
            response = session.get(self.login_url)

            soup = BeautifulSoup(response.text, "html.parser")
            csrf_token_tag = soup.find('input', {'name': 'csrfmiddlewaretoken'})
            if not csrf_token_tag:
                logging.error("csrf token not found... Continuing without csrf")
            else:
                self.payload["csrfmiddlewaretoken"] = csrf_token_tag.get("value", "")

                login_response = session.post(self.login_url, data=self.payload, headers=self.headers)

                if login_response.status_code == 200:
                    logging.info(f"Successfully logged in to the site {self.login_url}")

                else:
                    logging.error(f"Logging failed, status code {login_response.status_code}")

                file_save_path = Path(self.save_path)
                file_save_path.mkdir(parents=True, exist_ok=True)

                for year in range(self.year_range[0], self.year_range[1]+1):
                    data_url = self.data_base_url.format(year=year)

                    try:
                        download_response = session.get(data_url, headers=self.headers)
                        filename=f"{year}.csv"
                        path = file_save_path/filename

                        with open(path, "wb") as file:
                            file.write(download_response.content)

                        logging.info(f"Data for the year {year} downloaded successfully to {path}")

                    except requests.exceptions.RequestException as e:
                        logging.error(f"Failed to download data for {year}: {e}")


                    except Exception as e:
                        raise CustomException(e, sys)
                    
        except Exception as e:
            raise CustomException(e, sys)
        print("Data Downloades successfully!!!")
        logging.info("Data Downloades successfully!!!")


def ingestion_pipeline():
    obj = DataIngestion()
    obj.initiate_scrapping()