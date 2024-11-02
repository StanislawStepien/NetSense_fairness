# Project files
from collections.abc import Iterable

import query as q
import build_network as bn
import cogsNet
# Other imports
import abc
import pandas as pd
import networkx as nx
import numpy as np
import combu  # MIT License Copyright (c) 2020 Takeru Saito
from six import StringIO
from IPython.display import Image
import pydotplus
# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz

# NetSense has 6 questions about different worldview-related issues:
questions = ["euthanasia", "fssocsec", "fswelfare", "jobguar", "marijuana", "toomucheqrights"]
# Here's short overview of the files from CoDING I'm using.
"""
#files from the CoDING's GitHub, each file had the highest accuracy out of the other tests. Their accuracies are:
'euthanasia':euthanasia-0.1-iter10.csv',
with acc: 0.555045871559633
'fssocsec':fssocsec-0.8-iter1.csv',
with acc: 0.6972477064220184
'fswelfare':fswelfare-0.6-iter3.csv',
with acc: 0.6410550458715596
'jobguar':jobguar-0.1-iter5.csv',
with acc: 0.569954128440367
'marijuana':marijuana-0.1-iter7.csv',
with acc: 0.5756880733944955
'toomucheqrights':toomucheqrights-0.7-iter1.csv',
with acc: 0.5653669724770642
"""
# Those are the files from CoDING that have the highest acc. for each of the consecutive questions.
best_file_per_question = {
    'euthanasia': '../CoDING_best_files/euthanasia-0.1-iter10.csv',
    'fssocsec': '../CoDING_best_files/fssocsec-0.8-iter1.csv',
    'fswelfare': '../CoDING_best_files/fswelfare-0.6-iter3.csv',
    'jobguar': '../CoDING_best_files/jobguar-0.1-iter5.csv',
    'marijuana': '../CoDING_best_files/marijuana-0.1-iter7.csv',
    'toomucheqrights': '../CoDING_best_files/toomucheqrights-0.7-iter1.csv'
}
# Number of the survey for which we are going to prepare a classifier system. Possible values are from 2 to 6.
# Don't put 1 in here, because it's like asking 'what interactions did we have before the 1st semester?'
# - answer is not that many...
SURVEY_NUMBER = 3
# This parameter is for deciding whether classifier should only take into the account
# BehavioralAll data from the past 1 semester. If false, it will take all the data until
# the time of survey from the SURVEY_NUMBER (line above).
using_data_from_only_the_previous_semester = True
# parse CoDING results into a dataframe
coding_results: dict[str, pd.DataFrame] = {}
for question in questions:
    coding_results[question] = pd.read_csv(best_file_per_question[question], sep=";")
    coding_results[question].SurveyDate = pd.to_datetime(coding_results[question].SurveyDate, unit='s')
    # Only taking into the account entries for the given semester
    coding_results[question] = coding_results[question][coding_results[question].SurveyNr == SURVEY_NUMBER]
    # Renaming CoDING's StudentID column to EgoID will make
    # later joining of dataframes for purposes of extracting features a bit quicker.
    coding_results[question] = coding_results[question].rename(columns={"StudentID": "EgoID"})

# Getting the data from BehavioralAll table
# which corresponds to the SURVEY_NUMBER & using_data_from_only_the_previous_semester parameters
file_path = "../behaviorallAll/"
file = f"BehavioralAll_S{SURVEY_NUMBER}_sinceStart_{min(str(not using_data_from_only_the_previous_semester))}.csv"
try:
    print(f"Loading data from {file} file.")
    df = pd.read_csv(file_path + file, parse_dates=["DateTime"])
    print(f'File {file} read successfully')
except FileNotFoundError:
    print(f"File {file} not found. Going to run the database query instead..")
    query = q.construct_query_for_semester(SURVEY_NUMBER, using_data_from_only_the_previous_semester)
    df = q.run_query_return_df(query, allow_self_links=False)
    print(f"Data successfully loaded. Saving it to {file} file.")
    df.to_csv(file_path + file, index=False)

# We're only taking into the account agents with IDs of length==5,
# because others were outsiders who did not fill in the Surveys
df = df[df.SenderID <= 99999]
df = df[df.SenderID >= 10000]
df = df[df.ReceiverID <= 99999]
df = df[df.ReceiverID >= 10000]

# TODO: let's add some agents here who filled in the survey but are not present in the BehaviorallAll records

# In this step a network is being built from the retrieved data.
G: nx.classes.graph.Graph = bn.build_network(df, directed=False)
bn.draw_graph_of_g(G, show=False)

# Getting values of CogsNet for the network
print('Calculating CogsNet weights for the network..')
# Cij - number of interactions between agents i & j
# Tij - time of the last interaction between agents i & j
# Wij - CogsNet's weight for the link between agents i & j - numpy ndarray
Cij, Tij, Wij, all_nodes, weights_adjacency_matrix = cogsNet.run_cogsnet(df)
print('CogsNet weights calculated successfully.')
# this foo creates and displays a heatmap adjacency plot
bn.draw_adjacency_heatmap(weights_adjacency_matrix, show=False)

# In this step we collect a number of topological features which are going to be used for the classifier
features = ["Node degree", "Sum of CogsNet", "Avg Neighbour's CogsNetSum", "Avg Neighbour Degree", "Degree-centrality",
            "Betweenness-centrality", "Pagerank"]


# This function returns 2-3 features more than the list above would suggest.
# This is because of my volatile decision-making process when it comes to what
# exactly should constitute a 'topology-based' approach... I'm calculating those, but they won't be used in the end.
def get_features(network: nx.classes.graph.Graph, matrix_of_number_of_interactions: np.ndarray,
                 matrix_of_cogsnet_weights: np.ndarray, matrix_of_dates_of_last_interaction: np.ndarray,
                 list_of_all_nodes: list[int]) -> dict[str:list]:
    # First part of this foo calculates the 'personal' measures for each of the agents
    (degrees, cogsnetSum, interactionSum, avgTimeSinceLastInteraction, centrality_degree_list,
     centrality_betweenness_list, pagerank_list) = [], [], [], [], [], [], []
    centrality_degree = nx.degree_centrality(network)
    centrality_betweenness = nx.betweenness_centrality(network, k=len(list_of_all_nodes) - 1)
    pagerank = nx.pagerank(network)
    for node in list_of_all_nodes:
        degrees.append(network.degree(node))
        curr_index = list_of_all_nodes.index(node)
        cogsnetSum.append(sum(matrix_of_cogsnet_weights[curr_index]))
        interactionSum.append(sum(matrix_of_number_of_interactions[curr_index]))
        centrality_degree_list.append(centrality_degree[node])
        centrality_betweenness_list.append(centrality_betweenness[node])
        pagerank_list.append(pagerank[node])
        # avgTimeSinceLastInteraction is not going to be used as a feature,
        # because it's not really topological in nature. I'm leaving it here for now, but it's not getting used later...
        tsum = 0
        for register in matrix_of_dates_of_last_interaction[curr_index]:
            if register != 0:
                register = register.timestamp()
                timeFromNow = pd.Timestamp.now().timestamp() - register
                tsum += timeFromNow
        avgTimeSinceLastInteraction.append(tsum / len(matrix_of_dates_of_last_interaction[curr_index]))
    # Second part of this foo calculates the aggregate measures for each agents' neighbours
    avgNeighboursCogsNetSums, avgNeighboursInteractionSums, avgNeighbourDegrees = [], [], []
    for node in list_of_all_nodes:
        neighbours = network.neighbors(node)
        cogsnet = 0
        interaction = 0
        degree = 0
        # Calculating first the sum of parameters for each of the node's neighbors
        for neighbour in neighbours:
            cogsnet += sum(matrix_of_cogsnet_weights[list_of_all_nodes.index(neighbour)])
            interaction += sum(matrix_of_number_of_interactions[list_of_all_nodes.index(neighbour)])
            degree += network.degree(neighbour)
        # Getting the average of each parameter 'per neighbor' of the node
        nodes_degree = degrees[list_of_all_nodes.index(node)]
        # noinspection PyTypeChecker
        avgNeighboursCogsNetSums.append(cogsnet / nodes_degree)
        # noinspection PyTypeChecker
        avgNeighboursInteractionSums.append(interaction / nodes_degree)
        # noinspection PyTypeChecker
        avgNeighbourDegrees.append(degree / nodes_degree)

    # Parsing data into a dict of lists,
    # where n-th entry in every list corresponds to the features_dict['EgoID'][n]'s agent.
    features_dict = {"EgoID": list_of_all_nodes, "Node degree": degrees, "Sum of CogsNet": cogsnetSum,
                     "Avg Neighbour's CogsNetSum": avgNeighboursCogsNetSums,
                     "Avg Neighbour Degree": avgNeighbourDegrees,
                     "Avg Neighbour's InteractionSum": avgNeighboursInteractionSums,
                     "Degree-centrality": centrality_degree_list, "Betweenness-centrality": centrality_betweenness_list,
                     "Pagerank": pagerank_list}
    return features_dict


features_for_training_dict = get_features(network=G, matrix_of_number_of_interactions=Cij,
                                          matrix_of_cogsnet_weights=Wij,
                                          matrix_of_dates_of_last_interaction=Tij, list_of_all_nodes=all_nodes)

dict_of_training_dfs: dict[str, pd.DataFrame] = {}
for question in questions:
    # Parsing CoDING results together with the features obtained from the previous step.
    dict_of_training_dfs[question] = pd.merge(coding_results[question],
                                              pd.DataFrame.from_dict(features_for_training_dict), on="EgoID",
                                              how="left", validate="many_to_many")
    # has_coding_predicted_the_label column is going to be 1 if CoDING has successfully predicted agent's opinion
    # and 0 otherwise
    dict_of_training_dfs[question]['has_coding_predicted_the_label'] = np.where(
        dict_of_training_dfs[question]['OpinionSim'] == dict_of_training_dfs[question]["OpinionSurvey"], 1, 0)
    dict_of_training_dfs[question] = dict_of_training_dfs[question].fillna(0)
    dict_of_training_dfs[question].to_csv("training.csv", index=False)
# This is the dictionary which lists all the tested Classifier models together with their parameters
# and ranges on which they are going to be tested
classifiers = {
    DecisionTreeClassifier: {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best", "random"],
                             "max_depth": range(1, 50), "max_features": ["sqrt", "log2", None],
                             "min_samples_leaf": range(1, 10)},
    RandomForestClassifier: {"n_estimators": range(2, 200), "criterion": ["gini", "entropy", "log_loss"],
                             "max_depth": range(1, 50), "min_samples_leaf": range(1, 10),
                             "max_features": ["sqrt", "log2", None], "bootstrap": [True, False]},
    ExtraTreeClassifier: {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best", "random"],
                          "max_depth": range(1, 50), "max_features": ["sqrt", "log2", None],
                          "min_samples_leaf": range(1, 10)},
    LogisticRegression: {"penalty": ["l1", "l2", 'elasticnet', None], "tol": [0.01, 0.001, 0.0001, 0.00001],
                         "fit_intercept": [True, False], "class_weight": ['balanced', None],
                         "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']},
    GaussianNB: {},
    KNeighborsClassifier: {"n_neighbors": range(1, 10), "weights": ["uniform", "distance"],
                           "algorithm": ["auto", "ball_tree", "kd_tree", "brute"], "p": range(1, 5)}
}
Number_of_runs_per_experiment = 100


def initialize_experiment_attributes(data, training_features, list_of_class_models, list_of_questions):
    if data is None:
        data = dict_of_training_dfs
    if training_features is None:
        training_features = features
    if list_of_class_models is None:
        list_of_class_models = classifiers
    if list_of_questions is None:
        list_of_questions = questions
    return data, training_features, list_of_class_models, list_of_questions


def experiment(list_of_questions: list[str] = None, data=None,
               list_of_class_models=None, n: int = Number_of_runs_per_experiment,
               training_features=None) -> pd.DataFrame:
    # Initialize the output dict
    results = {'question': [], 'model': [], 'params': [], "accuracy": []}
    # Initialize mutable parameters
    data, training_features, list_of_class_models, list_of_questions = initialize_experiment_attributes(data,
                                                                                                        training_features,
                                                                                                        list_of_class_models,
                                                                                                        list_of_questions)
    # Iterate over all 6 questions
    for survey_question in list_of_questions:
        X = data[survey_question][training_features]
        Y = data[survey_question]['has_coding_predicted_the_label']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
        best_accuracy: float = 0.0
        for model, classifier_attributes in list_of_class_models.items():
            model_parametrizer = combu.Combu(model, progress=True)
            for result, parameters in model_parametrizer.execute(classifier_attributes):
                acc = 0
                for _ in range(n):
                    result = result.fit(X_train, y_train)
                    # Predict the response for test dataset
                    y_pred = result.predict(X_test)
                    # Model Accuracy, how often is the classifier correct?
                    acc += metrics.accuracy_score(y_test, y_pred)
                acc /= n
                results['question'].append(survey_question)
                results['model'].append(model)
                results['params'].append(parameters)
                results['accuracy'].append(acc)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = result
                    best_params = parameters
                    print(
                        f"For question: {survey_question}, best model was: {best_model} of parameters:{best_params}, with accuracy {best_accuracy}")
    results = pd.DataFrame.from_dict(results)
    return results


experiment_out_df = experiment(list_of_questions=questions, data=dict_of_training_dfs, n=Number_of_runs_per_experiment,
                               training_features=features, list_of_class_models=classifiers)
experiment_out_df.to_csv("experiment_out.csv", index=False)