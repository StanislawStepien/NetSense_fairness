import pandas as pd
import networkx as nx
import build_network as bn
import cogsNet
import numpy as np
import query as q
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

path = "C:\\Users\\staszek\\coding_verification\\pythonProject\\codingfiles\\"
filename = f"{path}euthanasia-0.1-iter10.csv"
coding_result = pd.read_csv(filename, sep=";")
coding_result.SurveyDate = pd.to_datetime(coding_result.SurveyDate, unit='s')
coding_s2 = coding_result[coding_result.SurveyNr == 2]

sql_query = """
DECLARE avg_survey_submission_date TIMESTAMP;
SET avg_survey_submission_date = (
  SELECT 
  timestamp_seconds(cast(avg(unix_seconds(ds2.completed_2)) as int64)) 
  FROM `netsense-411221.NetSense.DemSurveyS2` as ds2
);

SELECT ba.DateTime, ba.EgoID,ba.SenderID,ba.ReceiverID,ba.EventType,ba.EventLength FROM `NetSense.BehavioralAll` as ba
JOIN `NetSense.DemSurveyS2` as ds2
ON ba.EgoID=ds2.egoid
WHERE ba.DateTime <= avg_survey_submission_date;
"""
# df = q.run_query_return_df(sql_query, allow_self_links=False)  # in the whole dataset there are 4454 self-links
df = pd.read_csv("CoDING_Analysis.csv", parse_dates=["DateTime"])
df = df[df.SenderID <= 99999]
df = df[df.SenderID >= 10000]
df = df[df.ReceiverID <= 99999]
df = df[df.ReceiverID >= 10000]
print(df.shape)
G = bn.build_network(df)
GN = bn.draw_graph_of_g(G, show=True)
Cij, Tij, Wij, all_nodes, weights_adjacency_matrix = cogsNet.run_cogsnet(df)
# this foo creates and displays a heatmap adjacency plot
bn.draw_adjacency_heatmap(weights_adjacency_matrix)

nodes = list(G.nodes)
degrees = [G.degree(node) for node in nodes]
cogsnetsum = []
interactionsum = []
avgTimeBetweenInteractions = []
for node in all_nodes:
    cogsnetsum.append(sum(Wij[all_nodes.index(node)]))
    interactionsum.append(sum(Cij[all_nodes.index(node)]))
    tsum = 0
    for register in Tij[all_nodes.index(node)]:
        if register != 0:
            register = register.timestamp()
            timeFromNow = pd.Timestamp.now().timestamp() - register
            tsum += timeFromNow
    avgTimeBetweenInteractions.append(tsum / len(Tij[all_nodes.index(node)]))

avgNeighboursCogsNetSums = []
avgNeighboursInteractionSums = []
avgNeighbourDegrees = []
centrality_degree = nx.degree_centrality(G)
centrality_degree_list = []
centrality_betweenness = nx.betweenness_centrality(G, k=215)
centrality_betweenness_list = []
pagerank = nx.pagerank(G)
pagerank_list = []
for node in all_nodes:
    neighbours = G.neighbors(node)
    cogsnet = 0
    interaction = 0
    degree = 0
    for neighbour in neighbours:
        cogsnet += Wij[all_nodes.index(node)][all_nodes.index(neighbour)]
        interaction += Cij[all_nodes.index(node)][all_nodes.index(neighbour)]
        degree += G.degree(neighbour)
    avgNeighboursCogsNetSums.append(cogsnet / G.degree(node))
    avgNeighboursInteractionSums.append(interaction / G.degree(node))
    avgNeighbourDegrees.append(degree / G.degree(node))
    centrality_degree_list.append(centrality_degree[node])
    centrality_betweenness_list.append(centrality_betweenness[node])
    pagerank_list.append(pagerank[node])

print(len(nodes), len(degrees), len(interactionsum), len(avgTimeBetweenInteractions), len(cogsnetsum),
      len(avgNeighboursCogsNetSums), len(avgNeighboursInteractionSums), len(avgNeighbourDegrees),
      len(centrality_degree_list), len(pagerank_list))
dict_for_df = {"EgoID": nodes, "Node degree": degrees, "Sum of CogsNet": cogsnetsum,
               "Avg Neighbour's CogsNetSum": avgNeighboursCogsNetSums, "Avg Neighbour Degree": avgNeighbourDegrees,
               "Avg Neighbour's InteractionSum": avgNeighboursInteractionSums,
               "Degree-centrality": centrality_degree_list, "Betweenness-centrality": centrality_betweenness_list,
               "Pagerank": pagerank_list}
future_hybrid_approach_possible_features = {"Number of Interactions": interactionsum,
                                            "Avg time from last interactions": avgTimeBetweenInteractions, }
# todo: add beetweeness closeness pagerank eigenvectorcentrality
training_df = pd.DataFrame.from_dict(dict_for_df)
coding_s2 = coding_s2.rename(columns={"StudentID": "EgoID"})
big_df = pd.merge(coding_s2, training_df, on="EgoID", how="left", validate="many_to_many")
big_df['predict_label'] = np.where(big_df['OpinionSim'] == big_df["OpinionSurvey"], 1, 0)
features = ["Node degree", "Sum of CogsNet", "Avg Neighbour's CogsNetSum", "Avg Neighbour Degree", "Degree-centrality",
            "Betweenness-centrality", "Pagerank"]
X = big_df[features]
y = big_df['predict_label']
n=20

def experiment():
    best_model = None
    best_setting = ""
    best = 0
    for max_depth in range(3, 50):
        print("Max depth:",max_depth)
        for criterion in ["gini", "entropy"]:
            print("Criterion:", criterion)
            for min_leaf in range(1, 30):
                print("min_leaf:", min_leaf)
                for splittering in ["best", "random"]:
                    print("splittering:", splittering)
                    for max_features in range(1, 7):
                        print("max_features:", max_features)
                        acc = 0
                        for i in range(n):
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
                            # Create Decision Tree classifer object
                            clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                                         min_samples_leaf=min_leaf, splitter=splittering,
                                                         max_features=max_features)
                            # Train Decision Tree Classifer
                            clf = clf.fit(X_train, y_train)
                            # Predict the response for test dataset
                            y_pred = clf.predict(X_test)
                            # Model Accuracy, how often is the classifier correct?
                            acc += metrics.accuracy_score(y_test, y_pred)
                        acc = acc / n
                        if best < acc:
                            best = acc
                            best_setting = f"parameters: max depth{max_depth}, criterion: {criterion}, min_leaf:{min_leaf}, splittering: {splittering}, max features:{max_features}"
                            best_model = clf
                            print(best_setting, "with acc:", best)
    return best_model, best_setting, best
    print(best_setting)
    print("Accuracy:", best)


def test_run():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(max_depth=29, criterion="entropy", min_samples_leaf=2, max_features=4,
                                 splitter="random")
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy", metrics.accuracy_score(y_test, y_pred))
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=features, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('guess_coding.png')
    Image(graph.create_png())


# test_run()


best_model, best_setting, best = experiment()
dot_data = StringIO()
export_graphviz(best_model, out_file=dot_data,filled=True, rounded=True,special_characters=True, feature_names=features, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('guess_coding.png')
Image(graph.create_png())
"""
ctr = 0
bctr = 0
for i in list(G.nodes):
    if len(str(i))==5:
        ctr+=1
    else:
        bctr+=1
coding_survey1 = coding_result[(coding_result.StudentID.notna()) & (coding_result.SurveyNr == 2)]
"""
