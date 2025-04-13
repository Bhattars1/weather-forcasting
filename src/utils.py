from sklearn.model_selection import train_test_split
import yaml
import torch


# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

def split_data(data, labels, test_size= args["test_size"]):
    # Convert to PyTorch tensors first
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(data_tensor, labels_tensor, shuffle=False, test_size=test_size)
    
    return X_train, X_test, y_train, y_test


