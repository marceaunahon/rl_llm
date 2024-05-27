import csv
import argparse
import numpy as np
import os

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
    "--dataset", default="ladder", type=str, help="Dataset to evaluate (low or high)"
)

parser.add_argument(
    "--model-name",
    default="google/flan-t5-large",
    type=str,
    help="Model to evalute --- see models.py for an overview of supported models",
)
parser.add_argument(
    "--character", default="conservation", type=str, help="Character to evaluate"
)
args = parser.parse_args()

################################################################################################
# SETUP
################################################################################################
path_results = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}/{args.character}/{args.model_name.replace('/', '_')}.csv"
path_table = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}/tables"

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
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if row[1] == scenario_id:
                if 'decision1' in row : decisions[0] += 1
                elif 'decision2' in row : decisions[1] += 1
                elif 'decision3' in row : decisions[2] += 1
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

################################################################################################
# RUN COMPUTATION AND STORE RESULTS
################################################################################################

if not os.path.exists(path_table):
    os.makedirs(path_table)

np.save(f"{path_table}/{args.character}_{args.model_name}.npy", llm_reward_table())