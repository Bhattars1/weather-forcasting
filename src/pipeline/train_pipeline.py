import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.components.data_ingestion import weather_data_ingestion_pipeline, rain_data_ingestion_pipeline
from src.components.data_preprocessing import preprocessing_pipeline
from src.components.train import training_pipeline


def pipeline():
    weather_data_ingestion_pipeline()
    rain_data_ingestion_pipeline()
    preprocessing_pipeline()
    training_pipeline()

if __name__ == "__main__":
    pipeline()