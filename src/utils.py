from sklearn.model_selection import train_test_split
import yaml

# Load config
with open("./src/components/config.yaml", "r") as file:
    args = yaml.safe_load(file)

def split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=args["test_size"])
    return X_test, X_test, y_train, y_test
        