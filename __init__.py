


import networkx as nx
import numpy as np
def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))  # next_graph涉及到的所有edge
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):  # 上一时刻图中是否有该节点
            edges_positive.append(e)  # positive的边
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)  # 负采样

    # 划分训练集，测试集，验证集
    # train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive,
    #                                                                         edges_negative, test_size=val_mask_fraction +test_mask_fraction)
    # val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos,
    #                                                                                 test_neg, test_size=test_mask_fractio \n /
    #                                                                                             (test_mask_fraction +val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg

def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):  # 采样和positive同等数量的边
        idx_i = np.random.randint(0, nodes_num)  # 随机选择i,j节点
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:  # 自连接
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):  # pos的边
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:  # 存在之前的数据
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg

edge1 = [[1,2],[1,4][1,3],[5,2],[5,4],[5,3],[2,4],[4,3],[5,6],[5,7],[6,7]]