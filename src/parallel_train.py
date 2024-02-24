import os
import pandas as pd
import numpy as np
import argparse
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

def init():

    parser = argparse.ArgumentParser()

    parser.add_argument("--drop_cols", type=str)
    parser.add_argument("--target_col", type=str)
    parser.add_argument("--date_col", type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--tracking_uri", type=str)

    args, _ = parser.parse_known_args()

    global target_col
    target_col = args.target_col

    global date_col
    date_col = args.date_col

    global model_folder
    model_folder = args.model_folder

    global drop_cols
    drop_cols = args.drop_cols.split(",")

    global tracking_uri
    tracking_uri = args.tracking_uri
    
    global mlflow_client
    mlflow_client = mlflow.MlflowClient()

    # Connect to mlflow tracking server
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.start_run()

def register_model(model_obj, model_name, model_folder, mini_batch_context):
    
    # Dump model
    print("Dumping model...")
    relative_path = os.path.join(
        model_folder,
        *list(str(i) for i in mini_batch_context.partition_key_value.values()),
    )

    if not os.path.exists(relative_path):
        os.makedirs(relative_path)

    # Register Model
    print(f"Model saved. Registering {model_name} to AML model registry...")
    mlflow.sklearn.log_model(sk_model=model_obj,
                             registered_model_name=model_name,
                             artifact_path=relative_path
                            )
    return

# Change this function to change the scoring method(s)
def score_model(model, X_test, y_test):
    preds = model.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test, preds))
    return score

def run(input_data, mini_batch_context):

    store = mini_batch_context.partition_key_value['Store']
    brand = mini_batch_context.partition_key_value['Brand']
    model_name = f"{store}_{brand}"
    model_description = f"GradientBoostingRegressor for store_brand = f{model_name}"
    print(f"Running train.py for...{model_name}")
    
    with mlflow.start_run(run_name=f"{brand}_{store}_job", nested=True) as train_run:
        
        mlflow.set_tags({"brand": f"{brand}", "store": f"{store}"})
        mlflow.sklearn.autolog()
        print("Mlflow sklearn autologging enabled")

        if not isinstance(input_data, pd.DataFrame):
            raise Exception("Not a valid DataFrame input.")

        if target_col not in input_data.columns:
            raise Exception("No target column found from input tabular data")
        elif date_col not in input_data.columns:
            raise Exception("No date column found from input tabular data")

        print(f"partition_key_value = {mini_batch_context.partition_key_value}")

        # data cleaning
        input_data[date_col] = pd.to_datetime(input_data[date_col])
        input_data = input_data.set_index(date_col).sort_index(ascending=True)
        input_data = input_data.assign(Week_Year=input_data.index.isocalendar().week.values)
        input_data = input_data.drop(columns=drop_cols, errors="ignore")

        # traning & evaluation
        features = input_data.columns.drop(target_col)

        X = input_data[features].values
        y = input_data[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=12, shuffle=False
        )

        # Fit challenger model
        challenger_model = GradientBoostingRegressor(random_state=12)
        challenger_model.fit(X_train, y_train)

        # Load Champion model
        try: 
            print(f"Loading model {model_name}:latest...")
            champion_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/latest")
       
        except mlflow.exceptions.MlflowException as e:
            
            print(f"Model {model_name} not found. Registering new model...")
            champion_model = None
            challenger_score = score_model(challenger_model, X_test, y_test)
            mlflow.log_metric("challenger_test_score", challenger_score)
            register_model(challenger_model, model_name, model_folder, mini_batch_context)
            print(f"Model {model_name} registered.")
            return []

           
        # Compare models
        print("Comparing model champion and challenger models...")
        challenger_score = score_model(challenger_model, X_test, y_test)
        mlflow.log_metric("challenger_test_score", challenger_score)
        champion_score = score_model(champion_model, X_test, y_test)
        mlflow.log_metric("champion_test_score", champion_score)


        if challenger_score < champion_score: # Change this scoring method as needed for different metrics
            print(f"Challenger model has lower RMSE: {challenger_score} < {champion_score}")
            register_model(challenger_model, model_name, model_folder, mini_batch_context)
            print(f"New model {model_name} registered.")
            return []
        else:
            print(f"Champion model has lower RMSE: {champion_score} < {challenger_score}")
            print(f"Passthrough champion model.")
            return []