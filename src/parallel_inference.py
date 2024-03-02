import os
import pandas as pd
import argparse
import mlflow

def init():

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_cols", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--date_col", type=str)
    parser.add_argument("--tracking_uri", type=str)

    args, _ = parser.parse_known_args()

    global drop_cols
    drop_cols = args.drop_cols.split(",")

    global output_dir
    output_dir = args.output_dir

    global date_col
    date_col = args.date_col

    global tracking_uri
    tracking_uri = args.tracking_uri

    global mlflow_client
    mlflow_client = mlflow.MlflowClient()

    # Connect to mlflow tracking server
    mlflow.set_tracking_uri(tracking_uri)


def run(input_data, mini_batch_context):

    store = mini_batch_context.partition_key_value['Store']
    brand = mini_batch_context.partition_key_value['Brand']
    model_name = f"{store}_{brand}"
    output_data = input_data.copy()
    print(f"Output dataframe schema: {output_data.columns}")

    try:
        # Load model (sklearn / mlflow)
        print(f"Loading model {model_name}:latest...")
        reg = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/latest")
    except mlflow.exceptions.MlflowException as e:
         print("No model found. Exiting...")
         return []

    # Prep Data
    input_data[date_col] = pd.to_datetime(input_data[date_col])
    input_data = input_data.set_index(date_col).sort_index(ascending=True)
    input_data = input_data.assign(Week_Year=input_data.index.isocalendar().week.values)

    X_test = input_data.drop(columns=drop_cols, errors="ignore")

    # Make prediction
    predictions = reg.predict(X_test)

    # Combine prediction with input_data
    output_data['predictions'] = predictions

    # Save predictions to output dir
    relative_path = os.path.join(output_dir + '/')

    print(f"Relative path: {relative_path}...")
    if not os.path.exists(relative_path):
            os.makedirs(relative_path)
    output_path = f"{relative_path}{model_name}.csv"
    
    print(f"Saving predictions to {output_path}...")
    output_data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}.")

    return []