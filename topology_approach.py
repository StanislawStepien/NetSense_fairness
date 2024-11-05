# Project files
import pickle
import query as q
import cogsNet as cN
import build_network as bn
import visualization as v
# Other imports
import pandas as pd
import networkx as nx
import numpy as np
import combu  # MIT License Copyright (c) 2020 Takeru Saito
# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.utils.tests.test_testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import LinAlgWarning

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
    'euthanasia': '../original/CoDING_best_files/euthanasia-0.1-iter10.csv',
    'fssocsec': '../original/CoDING_best_files/fssocsec-0.8-iter1.csv',
    'fswelfare': '../original/CoDING_best_files/fswelfare-0.6-iter3.csv',
    'jobguar': '../original/CoDING_best_files/jobguar-0.1-iter5.csv',
    'marijuana': '../original/CoDING_best_files/marijuana-0.1-iter7.csv',
    'toomucheqrights': '../original/CoDING_best_files/toomucheqrights-0.7-iter1.csv'
}
# Number of the survey for which we are going to prepare a classifier system. Possible values are from 2 to 6.
# Don't put 1 in here, because it's like asking 'what interactions did we have before the 1st semester?'
# - answer is not that many...
SURVEY_NUMBER = 2
# This parameter is for deciding whether classifier should only take into the account
# BehavioralAll data from the past 1 semester. If false, it will take all the data until
# the time of survey from the SURVEY_NUMBER (line above).
# So simply put: if False then more data from BehavioralAll is considered
using_data_from_only_the_previous_semester = False
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
file_path = "../original/behavioralAll/"
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

# In this step a network is being built from the retrieved data.
G: nx.classes.graph.Graph = bn.build_network(df, directed=False)
demographic_data_path = "../original/data/demographicData.csv"
try:
    print(f"Loading data from {demographic_data_path} file.")
    demog_df = pd.read_csv(demographic_data_path)
    print(f'File {demographic_data_path} read successfully')
except FileNotFoundError:
    print(f"File {demographic_data_path} not found. Going to run the database query instead..")
    demog_df = q.get_demographic_df()
    print(f"Data successfully loaded. Saving it to {file} file.")
    demog_df.to_csv(demographic_data_path, index=False)

G = bn.add_agents_to_network_from_df(G, demog_df)
print("Generating an image")
bn.draw_graph_of_g(G, show=True)
# Getting values of CogsNet for the network
print('Calculating CogsNet weights for the network..')
# Cij - number of interactions between agents i & j
# Tij - time of the last interaction between agents i & j
# Wij - CogsNet's weight for the link between agents i & j - numpy ndarray
Cij, Tij, Wij, all_nodes, weights_adjacency_matrix = cN.run_cogsnet(df)
print('CogsNet weights calculated successfully.')
# this foo creates and displays a heatmap adjacency plot
bn.draw_adjacency_heatmap(weights_adjacency_matrix, show=True)

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
    LogisticRegression: {"penalty": ["l1", "l2", 'elasticnet', None], "tol": [0.01, 0.001, 0.0001, 0.00001],
                         "fit_intercept": [True, False], "class_weight": ['balanced', None],
                         "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']},
    DecisionTreeClassifier: {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best", "random"],
                             "max_depth": [1, 25, 50], "max_features": ["sqrt", "log2", None],
                             "min_samples_leaf": [1, 5, 10]},
    RandomForestClassifier: {"n_estimators": [200], "criterion": ["gini", "entropy", "log_loss"],
                             "max_depth": [1, 25, 50], "min_samples_leaf": [1, 5, 10],
                             "max_features": ["sqrt", "log2", None], "bootstrap": [True, False]},
    ExtraTreeClassifier: {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best", "random"],
                          "max_depth": [50], "max_features": ["sqrt", "log2", None],
                          "min_samples_leaf": [1, 5, 10]},
    GaussianNB: {},
    KNeighborsClassifier: {"n_neighbors": [1, 5, 10, 15, 20], "weights": ["uniform", "distance"],
                           "algorithm": ["auto", "ball_tree", "kd_tree", "brute"], "p": [1, 3, 5]}
}
Number_of_runs_per_experiment = 3


# It's said to be a good practice to not include mutable objects as default parameters for the function,
# so decided to do it this way, in this foo, so it doesn't add 10 lines to already long experiment foo.
def initialize_attributes(data: dict[pd.DataFrame], training_features: list[str], list_of_class_models: dict,
                          list_of_questions: list[str]) -> tuple[
    dict[pd.DataFrame], list[str], dict, list[str]]:
    if data is None:
        data = dict_of_training_dfs
    if training_features is None:
        training_features = features
    if list_of_class_models is None:
        list_of_class_models = classifiers
    if list_of_questions is None:
        list_of_questions = questions
    return data, training_features, list_of_class_models, list_of_questions


# In case of the LogisticRegression model, there is a set of combinations of solver/penalty foo which
# are not compatible. This foo checks for Combo combinations which contain such pairs and skips them.
# For more information of which pairs are possible, please refer to the LogisticRegression's documentation:
# https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
def skip_illegal_penalty_solver_pairs(
        result: DecisionTreeClassifier or ExtraTreeClassifier or RandomForestClassifier or LogisticRegression or GaussianNB or KNeighborsClassifier,
        parameters: dict) -> bool:
    if result.__class__ == LogisticRegression:
        if ((parameters["solver"] == "lbfgs" or
             parameters["solver"] == "newton-cg" or
             parameters["solver"] == "newton-cholesky" or
             parameters["solver"] == "sag")
                and parameters["penalty"] not in ["l2", None]):
            return True
        if parameters["solver"] == "liblinear" and parameters["penalty"] not in ["l2", "l1"]:
            return True
        if parameters["solver"] == "saga" and parameters["penalty"] not in ["l1", "l2", "elasticnet", None]:
            return True
    if parameters.get("penalty", 0) == "elasticnet":
        result.l1_ratio = 0.5
    return False


# Checks if the currently tested set of parameters does better than anything beforehand
# and updates the values if it does.
def check_if_acc_is_best_and_update_output(acc: float, f1_score: float, best_accuracy: float, best_f1_score: float,
                                           best_model: DecisionTreeClassifier or ExtraTreeClassifier or RandomForestClassifier or LogisticRegression or GaussianNB or KNeighborsClassifier,
                                           best_params: dict, result, parameters: dict,
                                           survey_question: str) -> tuple:
    if acc > best_accuracy:
        best_accuracy = acc
    if f1_score >= best_f1_score:
        # if curr f1 score is identical as the previous one, but has greater acc, then it will be taken as the new best.
        if f1_score == best_f1_score and acc == best_accuracy:
            best_f1_score = f1_score
            best_model = result
            best_params = parameters
        else:
            best_f1_score = f1_score
            best_accuracy = acc
            best_model = result
            best_params = parameters
        print(f"For question: {survey_question}, best model was: {best_model} "
              f"of parameters:{best_params}, with accuracy {best_accuracy} and f1 score {best_f1_score}")
    return best_model, best_params, best_accuracy, best_f1_score


# Foo for saving pickles of the best models from the experiment
def pickle_the_best_model(
        best_model: DecisionTreeClassifier or ExtraTreeClassifier or RandomForestClassifier or LogisticRegression or GaussianNB or KNeighborsClassifier,
        experiment_question: str) -> str:
    with open(
            f"../original/best_models/best_model_{experiment_question}"
            f"_S{SURVEY_NUMBER}_All{str(not using_data_from_only_the_previous_semester)}.pkl",
            'wb') as filename:
        # noinspection PyTypeChecker
        pickle.dump(best_model, filename)
    return f"best_model_{experiment_question}S{SURVEY_NUMBER}_All{str(not using_data_from_only_the_previous_semester)}.pkl"


# Visualize the best models for each of the questions
def get_img_filename(filename: str) -> str:
    # initializing substrings
    sub1 = "best_model_"
    sub2 = "_S"
    # getting index of substrings
    idx1 = filename.index(sub1)
    idx2 = filename.index(sub2)
    curr_question = ''
    # getting elements in between
    for idx in range(idx1 + len(sub1) + 1, idx2):
        curr_question = curr_question + filename[idx]
    return f"Best_model_image_{curr_question}_S{SURVEY_NUMBER}_All{str(not using_data_from_only_the_previous_semester)}.png"


def visualize_output(pickled_name, X_train, y_train) -> None:
    read_directory = "../original/best_models/"
    save_directory = "../original/images_best_models/"
    img_filename = get_img_filename(pickled_name)
    model = pickle.load(open(read_directory + pickled_name, 'rb'))
    v.visualize_model(model, save_directory, img_filename, features, X_train, y_train)


# Experiment saves the best classifier models as pickles and
# returns a pd.Dataframe with results for all tested algorithms/parameters together with their accuracy.
def repeat_fitting_n_times_get_acc_and_f1(n: int, acc: float, f1_score: float, X_train: pd.DataFrame,
                                          X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                                          result: DecisionTreeClassifier or ExtraTreeClassifier or RandomForestClassifier or LogisticRegression or GaussianNB or KNeighborsClassifier):
    for _ in range(n):
        result = result.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = result.predict(X_test)
        # Model Accuracy, how often is the classifier correct?
        acc += metrics.accuracy_score(y_test, y_pred)
        f1_score += metrics.f1_score(y_test, y_pred)
    acc /= n
    f1_score /= n
    return acc, f1_score


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=LinAlgWarning)
def experiment(list_of_questions: list[str] = None, data=None,
               list_of_class_models=None, n: int = Number_of_runs_per_experiment,
               training_features=None, generate_best_models_visualizations: bool = False) -> pd.DataFrame:
    # Initialize the output dict
    results = {'question': [], 'model': [], 'params': [], "accuracy": [], "f1_score": []}
    # Initialize mutable parameters
    data, training_features, list_of_class_models, list_of_questions = initialize_attributes(data, training_features,
                                                                                             list_of_class_models,
                                                                                             list_of_questions)
    # Iterate over all 6 questions
    for survey_question in list_of_questions:
        # noinspection PyTypeChecker
        X = data[survey_question][training_features]
        # noinspection PyTypeChecker
        Y = data[survey_question]['has_coding_predicted_the_label']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
        best_accuracy: float = 0.0
        best_f1_score: float = 0.0
        best_model, best_params = None, None
        for model, classifier_attributes in list_of_class_models.items():
            model_parametrizer = combu.Combu(model, progress=True)
            for result, parameters in model_parametrizer.execute(classifier_attributes):
                acc = 0
                f1_score = 0
                if skip_illegal_penalty_solver_pairs(result, parameters):
                    continue
                # Getting the average accuracy of the model after N runs
                acc, f1_score = repeat_fitting_n_times_get_acc_and_f1(n, acc, f1_score, X_train, X_test, y_train,
                                                                      y_test, result)
                results['question'].append(survey_question)
                results['model'].append(model)
                results['params'].append(parameters)
                results['accuracy'].append(acc)
                results['f1_score'].append(f1_score)
                # After testing model N (default 100) times, check if its accuracy is the best so far.
                (best_model, best_params,
                 best_accuracy, best_f1_score) = check_if_acc_is_best_and_update_output(acc, f1_score, best_accuracy,
                                                                                        best_f1_score, best_model,
                                                                                        best_params, result, parameters,
                                                                                        survey_question)
        pickle_name = pickle_the_best_model(best_model, survey_question)
        if generate_best_models_visualizations:
            visualize_output(pickle_name, X_train, y_train)
    results = pd.DataFrame.from_dict(results)
    return results


experiment_out_df = experiment(list_of_questions=questions, data=dict_of_training_dfs, n=Number_of_runs_per_experiment,
                               training_features=features, list_of_class_models=classifiers,
                               generate_best_models_visualizations=True)
# Save results of the experiment into a .csv file. Names will vary depending on whether program run included
# all BehavioralAll records until the SURVEY_NUMBER date or relayed solely on the specified semester's records.
experiment_out_df.to_csv(
    f"../original/experiment_results/experiment_out_S{SURVEY_NUMBER}_ALL{str(not using_data_from_only_the_previous_semester)}.csv",
    index=False)
