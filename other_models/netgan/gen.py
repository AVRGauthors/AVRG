import pickle

import other_models.netgan.netgan.utils as utils
from sys import argv
import networkx as nx
import os
import numpy as np


def generate(scores, tg_sum, num_graphs, name, orig_vals_dict=None):
    graphs = []
    for i in range(num_graphs):
        sparse_mat = utils.graph_from_scores(scores, tg_sum)
        if isinstance(sparse_mat, np.ndarray):
            g = nx.from_numpy_array(sparse_mat, create_using=nx.Graph())
        else:
            g = nx.from_scipy_sparse_matrix(sparse_mat, create_using=nx.Graph())
        g.name = f'{name}-NetGAN'
        if orig_vals_dict is not None:
            nx.set_node_attributes(g, name='value', values=orig_vals_dict)
        graphs.append(g)
    return graphs

# def generate(scores, tg_sum, name, num_graphs):
#     graphs = []
#     for i in range(num_graphs):
#         sparse_mat = utils.graph_from_scores(scores, tg_sum)
#         g = nx.from_scipy_sparse_matrix(sparse_mat, create_using=nx.Graph())
#         g.name = f'{name}-NetGAN'
#         graphs.append(g)
#     return graphs


def main():
    if len(argv) < 4:
        print('Needs gname, path to pickle scores and tg_sum, number of graphs')
        exit(1)

    gname, path = argv[1: 3]
    num_graphs = int(argv[3])

    scores, tg_sum = utils.load_pickle(path)
    graphs = generate(scores, tg_sum, num_graphs)

    os.makedirs('./src/netgan/dumps', exist_ok=True)
    pickle.dump(graphs, open(f'./src/netgan/dumps/{gname}_graphs.pkl.gz', 'wb'))
    return


if __name__ == '__main__':
    main()