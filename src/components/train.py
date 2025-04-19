import sys
import pickle
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib


import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.logger import logging
from src.exception_handler import CustomException
from src.utils import split_data
from src.components.algorithms import WindSpeedPredictionLSTM, HumidityMLP

# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

with open('./data/preprocessed_data/data.pkl', 'rb') as f:
    X = pickle.load(f)
with open('./data/preprocessed_data/labels.pkl', 'rb') as f:
    Y = pickle.load(f)

X_train, X_test, y_train, y_test = split_data(X, Y)



class WeatherForcastingTraining:
    def __init__(self, features=X_train, targets=y_train, test_features = X_test, test_targets = y_test):
        self.features = features
        self.targets = targets
        self.test_features = test_features
        self.test_targets = test_targets

    def train_precipitation_model(self):
        print("Training precipation model")
        logging.info("Training of  precipation forcasting model initiated")
        try:
            precp_features = self.features[:,:,4:].numpy().reshape(self.features[:,:,4:].shape[0], -1)
            precp_test_features = self.test_features[:,:,4:].numpy().reshape(self.test_features[:,:,4:].shape[0], -1)
            randonforest_regressor = RandomForestRegressor(n_estimators=150)
            randonforest_regressor.fit(precp_features, self.targets[:,:,4].squeeze().numpy())

            joblib.dump(randonforest_regressor, f"{args['models_save_path']}/{args['precipitation_model_filename']}")
            logging.info("Training of precipation model successful... Starting Evaluation")

            preds = randonforest_regressor.predict(precp_test_features)
            
            mse = mean_squared_error(self.test_targets[:,:,4].squeeze().numpy(), preds)
            mae = mean_absolute_error(self.test_targets[:,:,4].squeeze().numpy(), preds)
            r_squared = r2_score(self.test_targets[:,:,4].squeeze().numpy(), preds)
            
            logging.info("Precipation forcasting model evaluated Successfully")
            logging.info(f"Mean Squared Error of Precipation: {mse:.3f}")
            logging.info(f"Mean absolute Error of Precipation: {mae:.3f}")
            logging.info(f"R-Squared score of Precipation: {r_squared:.3f}")
        except Exception as e:
            raise CustomException(e,sys)       
    def train_temprature_model(self):
        logging.info("Training of temperation forcasting model initiated...")
        print("Training Temperature model")
        try:
            # Convert data into XGBoost DMatrix format
            temp_features = self.features[:,:,2:].reshape(self.features[:,:,2:].shape[0],-1)
            temp_test_features = self.test_features[:,:,2:].reshape(self.test_features[:,:,2:].shape[0],-1)
            dtrain = xgb.DMatrix(temp_features, self.targets[:,:,2])
            dtest = xgb.DMatrix(temp_test_features, self.test_targets[:,:,2])

            params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.03,
                "max_depth": 10
            }
    
            xgb_model = xgb.train(params, dtrain, num_boost_round = 125, callbacks=[xgb.callback.LearningRateScheduler(lambda epoch: 0.1 * (0.99 ** epoch))])
            xgb_model.save_model(f"{args['models_save_path']}/{args['temperature_model_filename']}")
            logging.info(f"Temperature model saved Successfully")
            logging.info("Training of temperature forcasting model successful !!! Starting Evaluation..")

            # Make predictions
            y_pred = xgb_model.predict(dtest)

            # Evaluate the model
            mse = mean_squared_error(self.test_targets[:,:,2], y_pred)
            mae = mean_absolute_error(self.test_targets[:,:,2], y_pred)
            r_squared= r2_score(self.test_targets[:,:,2], y_pred)

            logging.info("Temperature forcasting model evaluated Successfully")
            logging.info(f"Mean Squared Error of temperature: {mse:.3f}")
            logging.info(f"Mean absolute Error of temperature: {mae:.3f}")
            logging.info(f"R-Squared score of temperature: {r_squared:.3f}")


        except Exception as e:
            raise CustomException(e,sys)
    def train_windspeed_model(self):
        print("Train windspeed model")
        logging.info("Training of wind speed forcasting model initiated")
        try:

            input_dim = self.features.shape[2]
            hidden_dim = 68
            output_dim = 1
            windspeed_model = WindSpeedPredictionLSTM(input_dim=input_dim,
                                                    hidden_dim=hidden_dim,
                                                    output_dim=output_dim,
                                                    num_layers=2,
                                                    dropout=0.2)

            train_dataset = TensorDataset(self.features, self.targets[:,:,1])
            test_dataset = TensorDataset(self.test_features, self.test_targets[:,:,1])

            train_loader = DataLoader(train_dataset,
                                    batch_size=32,
                                    shuffle=True)
            test_loader = DataLoader(test_dataset,
                                    batch_size=32,
                                    shuffle=False)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(windspeed_model.parameters(), lr=0.001)

            epochs = 12
            logging.info("Wind speed Model training started.....")
            for epoch in range(epochs):
                print(f"Epoch {epoch+1} of {epochs}")
                windspeed_model.train()
                train_loss = 0.0

                for X_batch, y_batch in tqdm(train_loader):
                    optimizer.zero_grad()
                    out = windspeed_model(X_batch)
                    loss = criterion(out, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)
                avg_train_loss = train_loss / len(train_loader.dataset)

                logging.info(f"The train loss in epoch {epoch+1} is :{avg_train_loss}")
            logging.info("Training Successful! Starting Evaluation....")
            torch.save(windspeed_model.state_dict(), f"{args['models_save_path']}/{args['wind_speed_model_filename']}")
            windspeed_model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    out = windspeed_model(X_batch)
                    loss = criterion(out, y_batch)
                    test_loss += loss.item() * X_batch.size(0)
                avg_test_loss = test_loss / len(test_loader.dataset)
            logging.info("Evaluation successful !!!")
            logging.info(f"Epoch {epoch+1:02} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        except Exception as e:
            raise CustomException(e,sys)      
    def train_humidity_model(self):
        print("Train humidity model")
        logging.info("Training of humidity forecasting model initiated")
        try:
            humidity_features = self.features[:,:,3].unsqueeze(2)
            hunidity_test_features = self.test_features[:,:,3].unsqueeze(2)
            
            input_dim = humidity_features.shape[1]
            hidden_dim = 128
            output_dim = 1
            dropout = 0.2

            humidity_model = HumidityMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout

            )

            train_dataset = TensorDataset(humidity_features, self.targets[:,:,3])
            test_dataset = TensorDataset(hunidity_test_features, self.test_targets[:,:,3])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(humidity_model.parameters(), lr=0.001)

            epochs = 30
            logging.info("Humidity Model training loop started.....")

            for epoch in range(epochs):
                humidity_model.train()
                train_loss = 0.0
                print(f"Epoch {epoch+1} of {epochs}")

                for X_batch, y_batch in tqdm(train_loader):
                    X_batch = X_batch.view(X_batch.size(0), -1)
                    optimizer.zero_grad()
                    out = humidity_model(X_batch)  
                    out = torch.clamp(out, max=100)
                    loss = criterion(out, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)

                avg_train_loss = train_loss / len(train_loader.dataset)
                logging.info(f"Train loss at epoch {epoch+1}: {avg_train_loss:.4f}")

            logging.info("Training Successful! Starting Evaluation....")
            torch.save(humidity_model.state_dict(), f"{args['models_save_path']}/{args['humidity_model_filename']}")

            humidity_model.eval()
            test_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.view(X_batch.size(0), -1)
                    out = humidity_model(X_batch)
                    out = torch.clamp(out, max=100)
                    loss = criterion(out, y_batch)
                    test_loss += loss.item() * X_batch.size(0)
                    all_preds.extend(out.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())

            avg_test_loss = test_loss / len(test_loader.dataset)
            all_preds = np.array(all_preds).flatten()
            all_targets = np.array(all_targets).flatten()

            mae = mean_absolute_error(all_targets, all_preds)
            r2 = r2_score(all_targets, all_preds)

            logging.info(f"Test MSE: {avg_test_loss:.4f}")
            logging.info(f"Test MAE: {mae:.4f}")
            logging.info(f"Test R squared Score: {r2:.4f}")

        except Exception as e:
            raise CustomException(e, sys)


def training_pipeline():
    obj = WeatherForcastingTraining()
    # obj.train_precipitation_model()
    obj.train_temprature_model()
    # obj.train_windspeed_model()
    # obj.train_humidity_model()
training_pipeline()
        
