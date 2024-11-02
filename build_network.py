import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def build_network(df, node_id='SenderID', directed=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    if node_id in df.columns:
        for index, row in df.iterrows():
            try:
                G[int(row[node_id])][int(row['ReceiverID'])]['weight'] += 1
            except KeyError:
                curr_weight = 0
                G.add_edge(int(row[node_id]), int(row['ReceiverID']), weight=curr_weight + 1)
    return G


def draw_graph_of_g(G, show):
    edges_colormap = [edge[2]['weight'] for edge in G.edges(data=True)]
    nx.draw_networkx(G, pos=nx.spring_layout(G, 1), with_labels=False, edge_color=edges_colormap)
    if show:
        plt.show()


def draw_adjacency_heatmap(adjacency_weights, show=True):
    sns.heatmap(adjacency_weights, annot=False, cmap=sns.color_palette("Spectral", as_cmap=True))
    if show:
        plt.show()
