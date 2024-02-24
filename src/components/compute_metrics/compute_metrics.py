import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error

def get_rmse(gt, preds):
    rmse = np.sqrt(mean_squared_error(gt, preds))
    return rmse

def get_mape(gt, preds):
    mape = np.mean(np.abs((gt - preds) / gt) * 100)
    return mape

parser = argparse.ArgumentParser()
parser.add_argument("--ground_truth", type=str)
parser.add_argument("--predictions", type=str)
parser.add_argument("--ground_truth_column_name", type=str)
parser.add_argument("--predictions_column_name", type=str)
parser.add_argument("--join_cols", type=str)
parser.add_argument("--metric_results", type=str)

args, _ = parser.parse_known_args()

try:
    print(f"Loading ground truth from {args.ground_truth}")
    ground_truth_df = pd.read_csv(args.ground_truth)
    print(f"Loading predictions from {args.predictions}")
    predictions_df = pd.read_csv(args.predictions)
except:
    raise Exception("Can not load input data as csv tabular data.")


predictions_column_name = args.predictions_column_name
ground_truth_column_name = args.ground_truth_column_name
print(f"Loaded column names: {predictions_column_name}, {ground_truth_column_name}")

join_cols = args.join_cols.split(",")

print(f"Joining ground truth and predictions on {join_cols}")
joined_df = pd.merge(ground_truth_df, predictions_df,  how='inner', on=join_cols)

print(f"Output df created.\n {joined_df.head(5)}")
gt = joined_df[ground_truth_column_name]
preds = joined_df[predictions_column_name]

unique_combinations = joined_df[['Brand', 'Store']].drop_duplicates()

output_df = pd.DataFrame(columns=['StartDate', 'EndDate','Brand', 'Store', 'rmse', 'mape'])
start_date = joined_df['WeekStarting'].min()
end_date = joined_df['WeekStarting'].max()


for item in list(unique_combinations.itertuples(index=False, name=None)):

    print(f"Calculating metrics for {item}")
    data_partition = joined_df.loc[(joined_df['Brand'] == item[0]) & (joined_df['Store'] == item[1])]

    gt = data_partition[ground_truth_column_name]
    preds = data_partition[predictions_column_name]

    # Calculate RMSE
    rmse = get_rmse(gt, preds)

    # Calculate MAPE
    mape = get_mape(gt, preds)

    output_row = {
                    'StartDate': start_date, 
                    'EndDate': end_date,
                    'Brand': item[0],
                    'Store': item[1],
                    'rmse': rmse,
                    'mape': mape
                }

    output_df = output_df.append(output_row, ignore_index=True)

# write out the metrics
path = os.path.join(args.metric_results, "metrics.csv")
print(f"Writing out metrics to {path}")
output_df.to_csv(path, index=False)
print("Compute metrics complete.")