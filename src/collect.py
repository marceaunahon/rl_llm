"""Response Collection: Aggregate Model Responses into a single csv per model"""

import os
import pickle
import argparse
import pandas as pd
import csv
import numpy as np

from src.config import PATH_RESULTS


################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Collecting Results")
parser.add_argument(
    "--experiment-name",
    default="test",
    type=str,
    help="Name of Experiment - used for logging",
)
parser.add_argument(
    "--dataset", default="ladder", type=str, help="Dataset to evaluate"
)

parser.add_argument(
    "--character", default="conservation", type=str, help="Character to evaluate"
)

args = parser.parse_args()


################################################################################################
# SETUP
################################################################################################
path_results = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}/{args.character}"
path_results_raw = path_results + "_raw"


################################################################################################
# RESPONSE COLLECTION
################################################################################################
# Collect all pickle result files
results = []
for path, subdirs, files in os.walk(path_results_raw):
    for name in files:
        if name.endswith(".pickle"):
            path_file = os.path.join(path, name)

        with open(path_file, "rb") as f:
            tmp = pickle.load(f)
            results.append(tmp)

df_results = pd.concat(results)

# Store one csv per model
if not os.path.exists(path_results):
    os.makedirs(path_results)

def action_likelihood(decision : int, scenario_id : str, path : str = path_results) -> float:
    """
    Calculate the likelihood of a specific decision given a scenario ID.

    Parameters:
    decision (int): The decision index (0 for action1, 1 for action2).
    scenario_id (str): The ID of the scenario.
    path (str): The path to the results file (default: path_results).

    Returns:
    float: The likelihood of the specified decision.

    """
    # decision = 0 for decision1, 1 for decision2, 2 for decision3
    path = path_results
    decisions = [0,0,0]
    try :
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if row[1] == scenario_id:
                    if 'decision1' in row : decisions[0] += 1
                    elif 'decision2' in row : decisions[1] += 1
                    elif 'decision3' in row : decisions[2] += 1
    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"Error: {e}")
    sum = np.sum(decisions)
    decisions = [d/sum for d in decisions]
    return decisions[decision]

def llm_reward_table() -> np.ndarray:
    """
    Compute the reward table.

    Returns:
    table (np.ndarray): The reward table with shape (3,) representing the rewards for each action.
    """
    table = np.zeros(3)
    table[0] = action_likelihood(0, 'Bad') * (-1) + action_likelihood(1, 'Bad') * 0 + action_likelihood(2, 'Bad') * 1 #reward for decision 1
    table[1] = action_likelihood(0, 'Neutral') * (-1) + action_likelihood(1, 'Neutral') * 0 + action_likelihood(2, 'Neutral') * 1 #reward for decision 2
    table[2] = action_likelihood(0, 'Good') * (-1) + action_likelihood(1, 'Good') * 0 + action_likelihood(2, 'Good') * 1 #reward for decision 3
    return table

for model_id in df_results["model_id"].unique():
    results_model = df_results.loc[df_results["model_id"] == model_id]
    results_model.to_csv(
        f"{path_results}/{model_id.split('/')[0]}_{model_id.split('/')[-1]}.csv"
    )
    np.save(f"{path_results}/{model_id.split('/')[0]}_{model_id.split('/')[-1]}.npy", llm_reward_table())
