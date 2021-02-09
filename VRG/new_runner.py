import logging
import math
import os
import sys;
from glob import glob
from os.path import join
import numpy as np


sys.path.extend(['~/tmp/Attributed-VRG-tmp', '~/tmp/Attributed-VRG-tmp/VRG', '../', '../../'])

from VRG.src.parallel import parallel_async
from VRG.src import LightMultiGraph
from VRG.src.Tree import create_tree, draw_tree, readjust_tree, tree_okay
from VRG.src.extract import VRGExtractor
from VRG.src.other_graph_models import cell, netgan, dc_sbm, agm_fcl_runner, get_graphs_from_models
from VRG.src.utils import nx_to_lmg, load_pickle, dump_pickle, get_graph_from_prob_matrix

from pathlib import Path
from time import time
from typing import Any, List, Union, Dict, Tuple

import networkx as nx
import seaborn as sns
from tqdm import tqdm
from anytree import RenderTree

from VRG.runner import get_graph, get_clustering, make_dirs, get_grammars, generate_graphs

sys.setrecursionlimit(1_000_000)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
# logging.basicConfig(level=logging.ERROR, format="%(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True
sns.set_style('white')


def get_machine_name_and_outdir():
    name, outdir = None, None

    name_path = Path.home().joinpath('.name')
    outdir_path = Path.home().joinpath('.vrg_path')

    if name_path.exists():
        with open(name_path) as fp:
            name = fp.readline().strip()

    if outdir_path.exists():
        with open(outdir_path) as fp:
            outdir = fp.readline().strip()
    return name, outdir


def make_dirs(outdir: str, name: str) -> None:
    """
    Make the necessary directories
    :param outdir:
    :param name:
    :return:
    """
    subdirs = ('grammars', 'graphs', 'trees', 'generators')

    for dir in subdirs:
        dir_path = os.path.join(outdir, dir)
        if not os.path.exists(dir_path):
            logging.error(f'Making directory: {dir_path}')
            os.makedirs(dir_path)
        dir_path = os.path.join(dir_path, name)
        if not os.path.exists(dir_path):
            logging.error(f'Making directory: {dir_path}')
            os.makedirs(dir_path, exist_ok=True)
    return


def main():
    machine_name, outdir = get_machine_name_and_outdir()
    names = ['karate', 'football', 'polbooks', 'wisconsin', 'texas', 'film', 'cornell',
             'cora', 'citeseer', 'airports', 'polblogs', 'chameleon', 'pubmed', 'squirrel']

    clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random',
                   'leadingeig', 'consensus'][: -1]

    for name in names:
        g, _ = get_graph(name, basedir=outdir)
        make_dirs(outdir=outdir, name=name)
        for clustering in clusterings:
            tree = load_pickle(join(outdir, 'output', 'trees', name, f'{clustering}_list.pkl'))
            if tree is None: continue
            root = create_tree(tree)
            faulty_tnodes = tree_okay(root=root, g=g)
            if faulty_tnodes > 0: print(f'{name}\t{clustering}\t{faulty_tnodes:,d} errors')

    return


def netgan_cell_runner(outdir, input_g, name, model, graphs_filename=None, models_filename=None,
                       write_pickle=True):
    try:
        from src.other_graph_models import netgan
    except ModuleNotFoundError:
        from VRG.src.other_graph_models import netgan
    if model == 'netgan':
        graphs = netgan(input_g=input_g, name=name, outdir=outdir, use_model_pickle=True,
                        graphs_path=graphs_filename, models_path=models_filename, write_pickle=write_pickle)
    else:
        graphs = cell(input_g=input_g, name=name, outdir=outdir, use_model_pickle=True,
                      graphs_path=graphs_filename, model_path=models_filename, write_pickle=write_pickle)
    return graphs


def autoencoders(outdir, name, model):
    model_path = join(outdir, 'output', 'other_models', 'autoencoders')
    # if not Path(model_path).exists():
    #     os.makedirs(model_path)
    model_path = join(model_path, f'{name}_{model}_mat.pkl')
    graphs_path = join(outdir, 'output', 'graphs', name, f'{model}_10.pkl')

    # if Path(graphs_path).exists():
    #     return
    #
    input_g, _ = get_graph(name, basedir=outdir)
    if Path(model_path).exists():
        thresh_mat = load_pickle(model_path)
        graphs = []
        ns, ms = [], []
        for _ in range(10):
            g = get_graph_from_prob_matrix(thresh_mat, thresh=0.5)
            nx.set_node_attributes(g, name='value', values=nx.get_node_attributes(input_g, 'value'))
            ns.append(g.order())
            ms.append(g.size())
            graphs.append(g)
        print('Avg n, m', np.round(np.mean(ns), 3), np.round(np.mean(ms), 3))
        dump_pickle(graphs, graphs_path)
        return

    from other_models.autoencoders.fit import fit_model

    _, thresh_mat = fit_model(g=input_g, model_name=model)

    dump_pickle(thresh_mat, model_path)
    return


def read_batched_graphs(basedir, name):
    input_graphs = load_pickle(join(basedir, 'input', f'{name}.graphs'))
    cleaned_graphs = []

    for i, g in enumerate(input_graphs):
        g.remove_edges_from(nx.selfloop_edges(g))
        if not nx.is_connected(g):
            nodes_lcc = max(nx.connected_components(g), key=len)
            g = g.subgraph(nodes_lcc).copy()
        g = nx.convert_node_labels_to_integers(g, label_attribute='orig_label')
        g.name = f'{name}_{i}'
        cleaned_graphs.append(g)

    return cleaned_graphs


def cabam_other_models():
    basedir = '/data/ssikdar/Attributed-VRG'
    cabam_graphs = read_batched_graphs(basedir=basedir, name='cabam')
    models = ['CL', 'AGM', 'SBM', 'DC-SBM', 'cell', 'netgan']  # 'gcn_ae', 'gcn_vae', 'linear_ae', 'linear_vae']  #

    for model in models:
        for i, input_g in enumerate(cabam_graphs):
            name = f'cabam_{i}'
            graphs_filename = join(basedir, 'output', 'graphs', 'cabam', f'{model}_10_{i}.pkl')
            try:
                if model in ('netgan', 'cell'):
                    print(f'Running {model!r} on {name!r}')
                    netgan_cell_runner(outdir=basedir, model=model, name=name, input_g=input_g, graphs_filename=graphs_filename)
                # elif 'ae' in model:
                #     autoencoders(outdir=basedir, name=name, model=model, graphs_filename=graphs_filename)
                elif model in ('SBM', 'DC-SBM', 'CL', 'AGM'):
                    get_graphs_from_models(input_graph=input_g, num_graphs=10, name=name, model=model, graphs_filename=graphs_filename,
                                           outdir=basedir)
            except Exception as e:
                print(name, model, e)

    return


def old_main():
    basedir = '/data/ssikdar/Attributed-VRG'
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'airports',
             'polblogs', 'film', 'chameleon', 'squirrel'][: -3]
    models = ['gcn_ae', 'gcn_vae']

    args = []

    for name in names:
        input_g, _ = get_graph(gname=name, basedir=basedir)
        for model in models:
            try:
                if model in ('netgan', 'cell'):
                    netgan_cell_runner(outdir=basedir, model=model, name=name, input_g=input_g)
                elif 'ae' in model:
                    autoencoders(outdir=basedir, name=name, model=model)
                elif model in ('SBM', 'DC-SBM', 'CL', 'AGM'):
                    graphs = get_graphs_from_models(input_graph=input_g, num_graphs=10, name=name, model=model, outdir=basedir)
                    print(graphs)
            except Exception as e:
                print(name, model, e)

    exit(0)
    # for name in names:
    #     input_g, _ = get_graph(gname=name, basedir=basedir)
    #     for model in models:
    #         try:
    #             if model in ('netgan', 'cell'):
    #                 netgan_cell_runner(outdir=basedir, model=model, name=name, input_g=input_g)
    #             elif 'ae' in model:
    #                 autoencoders(outdir=basedir, name=name, model=model)
    #             elif model in ('SBM', 'DC-SBM', 'CL', 'AGM'):
    #                 get_graphs_from_models(input_graph=input_g, num_graphs=10, name=name, model=model, outdir=basedir)
    #         except Exception as e:
    #             print(name, model, e)
    exit(0)

    # for name in names[: ]:
    #     input_graph, attr_name = get_graph(name, basedir=outdir)
    #     for model in 'SBM', 'DC-SBM', 'CL', 'AGM':
    #         try:
    #             get_graphs_from_models(input_graph=input_graph, num_graphs=10, name=name, model=model, outdir=outdir)
    #         except Exception as e:
    #             print(e)

    exit(0)
    #
    # name = 'polblogs'
    # clustering = 'leiden'
    # grammar_type = f'AVRG-greedy-0'
    # mu = 5
    # num_graphs = 10
    #
    # grammar = load_pickle(glob(f'{outdir}/output/grammars/{name}/*AVRG*')[0])
    # graphs_filename = f'{outdir}/output/graphs/{name}/{grammar_type}_{clustering}_{mu}_{num_graphs}.pkl'
    # args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
    #              inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))
    #
    # exit(1)

    # autoencoders(outdir=outdir, name=name)
    # for name in names[1: -4]:
    #     netgan_cell_runner(outdir, name)
    # exit(1)

    # main()
    # Extracting grammar name:lang-bip mu:3 type:all_tnodes clustering:leiden
    # mu = 3
    # grammar_type = 'AVRG', 'all_tnodes'
    # clustering = 'leiden'
    # input_graph, attr_name = get_graph(name, basedir=outdir)
    # print(input_graph)
    # vrg = get_grammars(name=name, attr_name=attr_name, clustering=clustering, grammar_type=grammar_type,
    #                    input_graph=input_graph, mu=mu, use_cluster_pickle=True, use_grammar_pickle=True,
    #                    outdir=outdir, )
    # print(vrg)


def avrg_link_pred():
    basedir = '/data/ssikdar/Attributed-VRG/'
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', ]
    for name in names:
        input_graph, _ = get_graph(name, basedir=basedir)
        extract_type = 'mu_random'
        mu = 5
        clustering = 'leiden'
        grammar_filename = join(basedir, 'output/grammars', name, f'NCE_mu-random_{clustering}_{mu}.pkl')
        nce = get_grammars(name=name, grammar_type='NCE', extract_type=extract_type, clustering=clustering,
                           attr_name='value', input_graph=input_graph, mu=mu, outdir=basedir, use_grammar_pickle=True,
                           use_cluster_pickle=False, grammar_filename=grammar_filename)[0]

        print(nce)
        # AVRG-regular_mu-random_louvain_8_10.pkl
        graphs_filename = join(basedir, 'output/graphs', name, f'NCE_mu-random_{clustering}_{mu}_10.pkl')
        nce_graphs = generate_graphs(basedir=basedir, extract_type=extract_type, gen_type='NCE', grammar=nce,
                                     graphs_filename=graphs_filename, name=name, num_graphs=10, use_pickle=True)

        for out_g in nce_graphs:
            print(f'n={out_g.order():,d}, m={out_g.size():,d}, {type(out_g)}')
        print()
    return

    return


if __name__ == '__main__':
    # cabam_other_models()
    # avrg_link_pred()
    old_main()
