import cogsNet
import query
import build_network


def main():
    sql_query = 'SELECT * FROM netsense-411221.NetSense.BehavioralAll ORDER BY BehavioralAll.DateTime LIMIT 100'
    df = query.run_query_return_df(sql_query, allow_self_links=False)  # in the whole dataset there are 4454 self-links

    G = build_network.build_network(df)
    build_network.draw_graph_of_g(G, show=True)

    Cij, Tij, Wij, all_nodes, weights_adjacency_matrix = cogsNet.run_cogsnet(df)
    #this foo creates and displays a heatmap adjacency plot
    build_network.draw_adjacency_heatmap(weights_adjacency_matrix)


if __name__ == '__main__':
    main()
