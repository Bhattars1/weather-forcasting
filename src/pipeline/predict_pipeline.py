import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import split_data
from src.components.train import get_model

# Initialize model
input_size = 8
hidden_size = 128
output_size = 1
num_layers = 1

trained_model_path = "./model/weather_forecasting_lstm.pth"

model = get_model()
model.load_state_dict(torch.load(trained_model_path, weights_only=True))

with open('./data/preprocessed_data/data.pkl', 'rb') as f:
    X = pickle.load(f)
with open('./data/preprocessed_data/labels.pkl', 'rb') as f:
    Y = pickle.load(f)
batch_size = 32


with torch.inference_mode():
    _, X_test, _, y_test = split_data(X, Y, test_size=0.99)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = [] 
    with tqdm(test_loader, desc="Evaluating", unit="batch") as pbar:
        for inputs, targets in pbar:
            # Make predictions
            outputs = model(inputs)
            
            # Collect all predictions and true values
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

            # Update progress bar description with current status
            pbar.set_postfix(batch_size=len(inputs))




save_path = "./data/model_outputs"
os.makedirs(save_path, exist_ok=True)
all_preds = np.concatenate(all_preds).tolist()
all_labels = np.concatenate(all_labels).tolist()
predictions = pd.DataFrame(all_preds)
actual_values = pd.DataFrame(all_labels)
predictions.to_csv(f"{save_path}/predictions.csv", index=False)
actual_values.to_csv(f"{save_path}/actual_values.csv", index=False)


