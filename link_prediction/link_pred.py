import logging
from os.path import join
from pathlib import Path
from typing import Tuple, List

import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, \
    accuracy_score, f1_score

import sys;
import pandas as pd

from VRG.new_runner import netgan_cell_runner
from VRG.src.Tree import create_tree
from VRG.src.extract import AVRGLinkExtractor

sys.path.extend(['../', '../../', '../../../'])

from VRG.runner import get_grammars, get_graph, make_dirs, generate_graphs, get_clustering
from VRG.src.VRG import NCE
from VRG.src.generate import GreedyGenerator, NCEGenerator, EnsureAllNodesGenerator, AttributedEnsureAllNodesGenerator
from VRG.src.utils import check_file_exists, load_pickle, dump_pickle, nx_to_lmg
from link_prediction.utils import sparse_to_tuple, make_plot, sigmoid


class LinkPrediction:
    """

    """
    METRICS = 'AUPR', 'AUROC', 'ACC', 'AP', 'F1'

    def __init__(self, input_graph: nx.Graph, test_valid_split: Tuple[float, float], outdir: str,
                 dataset: str, use_pickle: bool = False, splits_filename=None):
        self.input_graph = input_graph
        self.outdir = outdir
        self.dataset = dataset
        self.method = None
        self.test_frac, self.valid_frac = test_valid_split
        self.splits_filename = splits_filename
        self.vals_dict = nx.get_node_attributes(self.input_graph, 'value')  # store the values of the nodes in the inp

        self.adj_train, self.train_edges, self.train_edges_false, self.val_edges, self.val_edges_false, \
            self.test_edges, self.test_edges_false = self._partition_graph(test_frac=self.test_frac,
                                                                           val_frac=self.valid_frac,
                                                                           verbose=True, use_pickle=use_pickle)

        self.performance = {metric: np.nan for metric in LinkPrediction.METRICS}
        return

    def set_method(self, method):
        self.method = method
        return

    def _partition_graph(self, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False, use_pickle=False):
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
        # taken from https://github.com/lucashu1/link-prediction/blob/master/gae/preprocessing.py
        if self.splits_filename is None:
            self.splits_filename = join(self.outdir, 'output', 'splits',
                                   f'{self.dataset}_{int(test_frac * 100)}_{int(val_frac * 100)}')
        if use_pickle and check_file_exists(self.splits_filename):
            logging.error(f'Using pickle at {splits_filename!r}')
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
                test_edges, test_edges_false = load_pickle(splits_filename)
        else:
            g = nx.Graph(self.input_graph)
            adj = nx.to_scipy_sparse_matrix(g)
            orig_num_cc = nx.number_connected_components(g)

            adj_triu = sp.triu(adj)  # upper triangular portion of adj matrix
            adj_tuple = sparse_to_tuple(adj_triu)  # (coords, values, shape), edges only 1 way
            edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
            # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
            num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
            num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be

            # Store edges in list of ordered tuples (node1, node2) where node1 < node2
            edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
            all_edge_tuples = set(edge_tuples)
            train_edges = set(edge_tuples)  # initialize train_edges to have all edges
            test_edges = set()
            val_edges = set()

            if verbose: print('generating test/val sets...', end=' ', flush=True)

            # Iterate over shuffled edges, add to train/val sets
            np.random.shuffle(edge_tuples)
            for edge in edge_tuples:
                node1, node2 = edge

                g.remove_edge(node1, node2)  # If removing edge would disconnect a connected component, backtrack
                if prevent_disconnect:
                    if nx.number_connected_components(g) > orig_num_cc:
                        g.add_edge(node1, node2)
                        continue

                # Fill test_edges first
                if len(test_edges) < num_test:
                    test_edges.add(edge)
                    train_edges.remove(edge)

                # Then, fill val_edges
                elif len(val_edges) < num_val:
                    val_edges.add(edge)
                    train_edges.remove(edge)

                # Both edge lists full --> break loop
                elif len(test_edges) == num_test and len(val_edges) == num_val: break

            if (len(val_edges) < num_val) or (len(test_edges) < num_test):
                print('WARNING: not enough removable edges to perform full train-test split!')
                print(f'Num. (test, val) edges requested: {num_test, num_val})')
                print(f'Num. (test, val) edges returned: {len(test_edges), len(val_edges)}')

            if prevent_disconnect: assert nx.number_connected_components(g) == orig_num_cc

            if verbose: print('creating false test edges...', end=' ', flush=True)

            test_edges_false = set()
            while len(test_edges_false) < num_test:
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])

                if idx_i == idx_j: continue

                false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

                # Make sure false_edge not an actual edge, and not a repeat
                if false_edge in all_edge_tuples: continue
                if false_edge in test_edges_false: continue

                test_edges_false.add(false_edge)

            if verbose: print('creating false val edges...', end=' ', flush=True)

            val_edges_false = set()
            while len(val_edges_false) < num_val:
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])

                if idx_i == idx_j: continue

                false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

                # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
                if false_edge in all_edge_tuples or \
                        false_edge in test_edges_false or \
                        false_edge in val_edges_false:
                    continue

                val_edges_false.add(false_edge)

            if verbose: print('creating false train edges...')

            train_edges_false = set()
            while len(train_edges_false) < len(train_edges):
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])

                if idx_i == idx_j: continue

                false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

                # Make sure false_edge in not an actual edge, not in test_edges_false,
                # not in val_edges_false, not a repeat
                if false_edge in all_edge_tuples or \
                        false_edge in test_edges_false or \
                        false_edge in val_edges_false or \
                        false_edge in train_edges_false:
                    continue

                train_edges_false.add(false_edge)

            if verbose: print('final checks for disjointness...', end=' ', flush=True)

            # assert: false_edges are actually false (not in all_edge_tuples)
            assert test_edges_false.isdisjoint(all_edge_tuples)
            assert val_edges_false.isdisjoint(all_edge_tuples)
            assert train_edges_false.isdisjoint(all_edge_tuples)

            # assert: test, val, train false edges disjoint
            assert test_edges_false.isdisjoint(val_edges_false)
            assert test_edges_false.isdisjoint(train_edges_false)
            assert val_edges_false.isdisjoint(train_edges_false)

            # assert: test, val, train positive edges disjoint
            assert val_edges.isdisjoint(train_edges)
            assert test_edges.isdisjoint(train_edges)
            assert val_edges.isdisjoint(test_edges)

            if verbose: print('creating adj_train...', end=' ', flush=True)

            # Re-build adj matrix using remaining graph
            adj_train = nx.adjacency_matrix(g)

            # Convert edge-lists to numpy arrays
            train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
            train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
            val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
            val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
            test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
            test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

            if verbose: print('Done with train-test split!')

            # NOTE: these edge lists only contain single direction of edge!
            dump_pickle((adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false), splits_filename)
        logging.error(f'train (T/F): {len(train_edges)} valid: {len(val_edges)} ({val_frac*100}%) test: {len(test_edges)} ({test_frac*100}%)')
        return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

    def evaluate(self, predicted_adj_mat: np.array, make_plots: bool = True):
        """
        Evaluate the performance of the link prediction algorithm with the generated graph
        fill the
        :return:
        """
        true_edges, false_edges = [], []
        y_score = []

        for i, (u, v) in enumerate(self.test_edges):
            score = predicted_adj_mat[u, v]
            y_score.append(score)  # if there's an edge - it'll be 1 or close to 1
            true_edges.append(1)  # actual edge

        for i, (u, v) in enumerate(self.test_edges_false):
            score = predicted_adj_mat[u, v]
            y_score.append(score)  # the numbers should be 0 or close to 0
            false_edges.append(0)  # actual non-edge

        y_true = true_edges + false_edges
        assert len(y_score) == len(y_true), f'Lengths of y_score: {len(y_score)!r} y_true: {len(y_true)!r} not equal'

        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
        auroc = roc_auc_score(y_true=y_true, y_score=y_score)

        prec, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
        aupr = auc(recall, prec)

        self.performance['AUROC'] = auroc
        self.performance['AUPR'] = aupr
        self.performance['AP'] = average_precision_score(y_true=y_true, y_score=y_score)
        # try:
        #     acc = accuracy_score(y_true=y_true, y_pred=y_score)
        # except Exception:
        #     acc = np.nan
        # self.performance['ACC'] = acc
        # self.performance['F1'] = f1_score(y_true=y_true, y_pred=y_score)

        if make_plots:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            fig.set_size_inches(10, 8)
            make_plot(x=fpr, y=tpr, xlabel='False Positive Rate', ylabel='True Positive Rate',
                      c='darkorange', label=f'AUROC {round(auroc, 2)}',
                      title=f'{self.method!r} {self.dataset!r} ROC curve', ax=ax1)

            make_plot(x=recall, y=prec, xlabel='Recall', ylabel='Precision', c='darkorange',
                      label=f'AUPR {round(aupr, 2)}',
                      title=f'{self.method!r} {self.dataset!r} PR curve', ax=ax2, kind='pr')
            plt.show()
        return

    def __str__(self):
        st = f'\n<dataset: {self.dataset!r} method: {self.method!r} test: {self.test_frac*100}% valid: {self.valid_frac*100}%'
        perf = {k: np.round(v, 3) for k, v in self.performance.items() if not np.isnan(v)}
        st += f' performance: {perf!r}>'
        return st


def combine_graphs_into_matrices(graphs: List[nx.Graph]) -> np.array:
    combined_adj_mat = None
    nodelist = sorted(graphs[0].nodes)
    for g in graphs:
        adj_mat = nx.adjacency_matrix(g, nodelist=nodelist)
        if combined_adj_mat is None:
            combined_adj_mat = adj_mat
        else:
            combined_adj_mat += adj_mat

    combined_adj_mat = combined_adj_mat / len(graphs)
    return combined_adj_mat


def avrg_runner(link_pred_obj: LinkPrediction, count: int, mu: int, basedir: str) -> np.array:
    """
    Runs AVRG on the input graph and returns the combined adjacency matrix
    """

    extract_type = 'mu_random'
    mu = 5
    clustering = 'leiden'

    link_pred.set_method(method=f'AVRG_link_{clustering}_{mu}')
    train_g: nx.Graph = nx.from_scipy_sparse_matrix(link_pred_obj.adj_train, create_using=nx.Graph)
    train_g.add_edges_from(link_pred_obj.val_edges.tolist())   # add the validation edges too
    nx.set_node_attributes(train_g, name='value', values=link_pred_obj.vals_dict)

    list_of_list_clusters = get_clustering(g=train_g, outdir=basedir,
                                           clustering=clustering, use_pickle=False,
                                           filename='na', write_pickle=False)

    root = create_tree(list_of_list_clusters) if isinstance(list_of_list_clusters, list) else list_of_list_clusters
    train_lmg = nx_to_lmg(nx_g=train_g)
    extractor = AVRGLinkExtractor(g=train_lmg, attr_name=att_name, clustering=clustering, mu=mu,
                                  extract_type=extract_type, root=root)

    avrg_link = extractor.extract()

    mix_dict = nx.attribute_mixing_dict(train_g, 'value')
    gen = AttributedEnsureAllNodesGenerator(grammar=avrg_link, attr_name='value', mixing_dict=mix_dict,
                                            use_fancy_rewiring=True)
    gen_graphs = gen.generate(10)
    return gen_graphs


def autoencoder_runner(model: str, link_pred_obj: LinkPrediction) -> np.array:
    from link_prediction.methods.autoencoders.linear_gae.gae_fit import fit_model
    link_pred.set_method(method=model)
    mat = fit_model(adj=link_pred_obj.adj_train, val_edges=link_pred_obj.val_edges,
                    val_edges_false=link_pred_obj.val_edges_false, test_edges=link_pred_obj.test_edges,
                    test_edges_false=link_pred_obj.test_edges_false, model_name=model)
    return mat


def basic_runner(link_pred_obj: LinkPrediction, kind: str) -> np.array:
    link_pred.set_method('Jaccard')
    train_g: nx.Graph = nx.from_scipy_sparse_matrix(link_pred_obj.adj_train, create_using=nx.Graph)
    train_g.add_edges_from(link_pred_obj.val_edges.tolist())  # add the validation edges too

    pred_mat = np.zeros((train_g.order(), train_g.order()))
    only_compute = link_pred_obj.test_edges.tolist() + link_pred_obj.test_edges_false.tolist()
    if kind == 'jaccard':
        func = nx.jaccard_coefficient
    elif kind == 'adamic-adar':
        func = nx.adamic_adar_index
    else:
        raise NotImplementedError()
    for u, v, d in func(train_g, only_compute):
        pred_mat[u, v] = d
    return pred_mat


if __name__ == '__main__':
    basedir = '/data/ssikdar/Attributed-VRG'
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'airports', 'polblogs', 'film', 'chameleon', 'squirrel'][: 5]
    models = ['AVRG', 'gcn_ae', 'gcn_vae', 'linear_ae', 'linear_vae', 'jaccard', 'adamic-adar']  # 'netgan', 'cell', ]
    # names = ['citeseer']
    models = ['cell']

    for name in names:
        name_fname = join(basedir, 'stats/link_pred', f'{name}.csv')
        orig_g, att_name = get_graph(name, basedir=basedir)
        model_dfs = []
        trials = 10
        test_frac, val_frac = 0.1, 0.05
        for model in models:
            model_rows = []
            model_fname = join(basedir, 'stats/link_pred', f'{name}_{model}.csv')
            if Path(model_fname).exists():
                model_df = load_pickle(model_fname)
                continue
            for trial in range(1, trials+1):
                splits_filename = join(basedir, 'output', 'splits',
                                       f'{name}_{int(test_frac*100)}_{int(val_frac*100)}_{trial}.pkl')

                link_pred = LinkPrediction(input_graph=orig_g, test_valid_split=(test_frac, val_frac),
                                           dataset=name, use_pickle=True, outdir=basedir,
                                           splits_filename=splits_filename)  # use a diff split each time
                if 'ae' in model:
                    pred_mat = autoencoder_runner(model=model, link_pred_obj=link_pred)
                elif 'AVRG' in model:
                    graphs = avrg_runner(link_pred_obj=link_pred, count=10, mu=5, basedir=basedir)
                    pred_mat = combine_graphs_into_matrices(graphs)
                elif model in ('netgan', 'cell'):
                    train_g = nx.from_scipy_sparse_matrix(link_pred.adj_train, create_using=nx.Graph)
                    graphs = netgan_cell_runner(outdir=basedir, name=f'{model}-{trial}', model=model,
                                                input_g=train_g, write_pickle=False)
                    pred_mat = combine_graphs_into_matrices(graphs)
                else:
                    pred_mat = basic_runner(link_pred_obj=link_pred, kind=model)

                link_pred.evaluate(predicted_adj_mat=pred_mat, make_plots=False)
                perf = link_pred.performance
                row = dict(name=name, model=model, trial=trial, **perf)
                model_rows.append(row)
            model_df = pd.DataFrame(model_rows)
            model_dfs.append(model_df)
            model_df.to_csv(model_fname, index=False)
        # link_pred_df = pd.concat(model_dfs)
        # link_pred_df.to_csv(name_fname, index=False)
    exit(1)
