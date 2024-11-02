import numpy as np
from math import e
import pandas as pd

mi, theta, lambda_parameter = 0.4, 0.2, 0.005631
pd.set_option("display.max_columns", None)


def initizalize_cogsnet_parameters(df):
    unique_receivers = df['ReceiverID'].unique().tolist()
    unique_senders = df['SenderID'].unique().tolist()
    all_ids = list(set(unique_receivers + unique_senders))  # by using list(set()) we drop the duplicates

    number_of_nodes = len(all_ids)
    Cij = np.zeros((number_of_nodes, number_of_nodes))  # holds the count of events processed for this pair of nodes
    Tij = np.zeros(
        (number_of_nodes, number_of_nodes)).astype(
        pd.Timestamp)  # represents the time of the most recent event for this pair of nodes
    Wij = np.zeros((number_of_nodes, number_of_nodes))  # weights of all edges, i.e., for all pairs of nodes
    return Cij, Tij, Wij, all_ids


def forgetting_exponential(delta_time):
    return e ** (-1 * lambda_parameter * delta_time)


def forgetting_power(delta_time):
    return max(1, delta_time) ** (-1 * lambda_parameter)


def calculate_weight(n1, n2, time, Cij, Tij, Wij, forgetting_foo, directed_net=False):
    curr_wij = Wij[n1, n2]
    last_time = Tij[n1][n2] if Tij[n1][n2] != 0 else time
    if curr_wij * forgetting_foo((time - last_time).total_seconds() / 60 / 60) < theta:
        new_wij = mi
    else:
        new_wij = mi + curr_wij * forgetting_foo((time - last_time).total_seconds() / 60 / 60) * (1 - mi)

    Cij[n1, n2] += 1
    Tij[n1, n2] = time
    Wij[n1, n2] = new_wij
    if not directed_net:
        Wij[n2, n1] = new_wij
        Cij[n2, n1] += 1
        Tij[n2, n1] = time


def get_updated_matrices_from_df(Cij, Tij, Wij, all_nodes, df):
    for index, row in df.iterrows():
        x, y = all_nodes.index(row['SenderID']), all_nodes.index(row['ReceiverID'])
        if x != y:
            # last parameter here determines which of the forgetting functions we're using
            calculate_weight(x, y, row['DateTime'], Cij, Tij, Wij, forgetting_power)
    # preparing an adjacency matrix
    weights_adjacency_matrix = pd.DataFrame(Wij, columns=all_nodes, index=all_nodes)
    return Cij, Tij, Wij, all_nodes, weights_adjacency_matrix


def run_cogsnet(df):
    Cij, Tij, Wij, all_nodes = initizalize_cogsnet_parameters(df)
    return get_updated_matrices_from_df(Cij, Tij, Wij, all_nodes, df)
