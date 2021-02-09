import logging
import random
import sys
import time
from glob import glob
from os.path import join
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np
import os

sys.path.extend(['/home/ssikdar/tmp_dir', '../', '../../', '../../../'])

from VRG.src.motif_counter import igraph_read_gml, MotifCounter
from VRG.runner import get_grammars, get_graph, get_clustering, generate_graphs, make_dirs
from VRG.src.VRG import AttributedVRG, VRG
from VRG.src.parallel import parallel_async
from VRG.src.utils import load_pickle, get_mixing_dict, timer, nx_to_igraph


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


def batch_cluster_shuffler_runner():
    shuffle_kind = 'edges'
    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random', 'consensus']
    clusterings = ['cond', 'leiden', 'louvain', 'leadingeig']
    use_pickle = True
    args = []

    # for graph_filename in glob(f'./input/shuffled/{shuffle_kind}/toy-comm-*.gexf'):
    shuffle_kind = 'attrs'
    for graph_filename in glob(f'./input/shuffled/{shuffle_kind}/toy-comm-0.gexf'):
        path = Path(graph_filename)
        g = nx.read_gexf(graph_filename, node_type=int)
        # name = f'{path.stem}-{shuffle_kind}'
        name = 'toy-comm-attr'
        g.name = name
        for clustering in clusterings:
            args.append((g, f'/data/ssikdar/attributed-vrg/dumps/trees/{name}', clustering, use_pickle))

    parallel_async(func=get_clustering, args=args)
    return


def batch_cluster_runner(names, outdir, clusterings=None):
    if clusterings is None:
        clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'infomap', 'labelprop', 'random',
                       'leading_eig', 'consensus'][: -1]
    use_pickle = True
    args = []

    for name in names:
        g, _ = get_graph(name, basedir=outdir)
        g.name = name
        for clustering in clusterings:
            args.append((g, outdir, clustering, use_pickle, ''))
    random.shuffle(args)
    parallel_async(func=get_clustering, args=args)
    return


def batch_grammar_runner(names, clusterings, outdir, mus=None, extract_types=None, num_workers=8, shuffle=False):
    # grammar_types_1 = ['VRG', 'AVRG']
    grammar_types = ['AVRG']
    if extract_types is None:
        extract_types = ['mu_random', 'mu_level', 'all_tnodes']
    if mus is None:
        mus = range(3, 11)
    # mus = [5, 6]
    use_cluster_pickle = True
    use_grammar_pickle = True
    count = 1
    args = []
    write_pickle = True

    for name in names:
        input_graph, attr_name = get_graph(name, basedir=outdir)

        for clustering in clusterings:
            for grammar_type in grammar_types:
                for extract_type in extract_types:
                    for mu in mus:
                        extract = extract_type.replace('_', '-')
                        if extract_type == 'all_tnodes':
                            mu = -1
                        grammar_filename = join(outdir, 'output', 'grammars', name,
                                                f'{grammar_type}_{extract}_{clustering}_{mu}.pkl')

                        arg = (name, grammar_type, extract_type, clustering, mu, input_graph, use_grammar_pickle,
                               use_cluster_pickle, attr_name, outdir, count, grammar_filename, write_pickle)
                        args.append(arg)
                        if extract_type == 'all_tnodes':  # here mu is not important for all_tnodes
                            break
    print(args[: 3])
    if shuffle:
        random.shuffle(args)
    try:
        parallel_async(func=get_grammars, args=args, num_workers=num_workers)
    except Exception as e:
        print(e)

    ## get_grammars(name: str,  grammar_type: str, extract_type: str, clustering: str, mu: int, input_graph: nx.Graph,
    # use_grammar_pickle: bool, use_cluster_pickle: bool, attr_name: str, outdir: str, count: int = 1,
    # grammar_filename: str = '', write_pickle: bool = True, list_of_list_clusters=None)
    ##

    return


def batch_generator_runner(names, basedir, clusterings, mus=None, extract_types=None,
                           save_snapshots=False, num_workers=10, shuffle=False):
    num_graphs = 10  # we need 1 graph to chart the progress  # TODO: change this in the future?
    use_pickle = True
    save_snapshots = save_snapshots
    if mus is None:
        mus = list(range(3, 11)) + [-1]
    alpha = None

    args = []
    for name in names:
        input_graph, attr_name = get_graph(name, basedir=basedir)
        if input_graph.size() > 3_000:
            save_snapshots = False

        mix_dict = get_mixing_dict(input_graph, attr_name=attr_name)
        inp_deg_ast = nx.degree_assortativity_coefficient(input_graph)
        inp_attr_ast = nx.attribute_assortativity_coefficient(input_graph, attr_name)

        for grammar_filename in glob(f'{basedir}/output/grammars/{name}/*'):
            grammar = load_pickle(grammar_filename)
            extract_type = grammar.extract_type.replace('_', '-')
            if grammar.mu not in mus or grammar.clustering not in clusterings or extract_type not in extract_types:
                continue
            print(Path(grammar_filename).stem)

            if isinstance(grammar, AttributedVRG):
                for gen_type, fancy in zip(('AVRG-regular', 'AVRG-fancy'), (False, True)):
                    graphs_filename = f'{basedir}/output/graphs/{name}/{gen_type}_{extract_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}.pkl'
                    args.append((name, grammar, num_graphs, extract_type, gen_type, basedir, graphs_filename, mix_dict,
                                 attr_name, fancy, inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha))

                for alpha, gen_type in zip((0, 0.5, 1), ('AVRG-greedy-attr', 'AVRG-greedy-50', 'AVRG-greedy-deg')):
                    fancy = None
                    graphs_filename = f'{basedir}/output/graphs/{name}/{gen_type}_{extract_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}.pkl'
                    args.append((name, grammar, num_graphs, extract_type, gen_type, basedir, graphs_filename,
                                 mix_dict, attr_name, fancy, inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha))

            else:
                continue  # skip VRGs
                # assert isinstance(grammar, VRG)
                # grammar_type = 'VRG'
                # fancy = None
                # graphs_filename = f'{basedir}/output/graphs/{name}/{grammar_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}.pkl'
                # args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                #              inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha, graphs_filename))
    if shuffle:
        random.shuffle(args)
    try:
        parallel_async(func=generate_graphs, args=args, num_workers=num_workers)
    except Exception as e:
        print(e)
    return


def batch_synthetic_generator_runner():
    # frac = np.linspace(0, 1, 21, endpoint=True) * 100
    frac = np.linspace(0, 100, 11, endpoint=True, dtype=int)  # change it to increments of 10 for now
    names = [f'toy-comm-{f}' for f in frac]
    # names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    num_graphs = 5
    outdir = '/data/ssikdar/attributed-vrg/dumps'
    use_pickle = True
    save_snapshots = False
    shuffle = 'edges'

    args = []
    for name in names:
        # input_graph, attr_name = get_graph(name)
        input_graph, attr_name = nx.read_gexf(f'./input/shuffled/{shuffle}/{name}.gexf', node_type=int), 'block'
        name = f'{name}-{shuffle}'
        if attr_name == '':
            mix_dict, inp_deg_ast, inp_attr_ast = None, None, None
        else:
            mix_dict = get_mixing_dict(input_graph, attr_name=attr_name)
            inp_deg_ast = nx.degree_assortativity_coefficient(input_graph)
            inp_attr_ast = nx.attribute_assortativity_coefficient(input_graph, attr_name)

        for grammar_filename in glob(f'{outdir}/grammars/{name}/*'):
            grammar = load_pickle(grammar_filename)
            if isinstance(grammar, AttributedVRG):
                grammar_type = 'AVRG'
                fancy = True
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))

                grammar_type = 'AVRG-greedy'
                # args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                #              inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))
                for alpha in (0, 0.5, 1):
                    args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                                 inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha))
            else:
                assert isinstance(grammar, VRG)
                grammar_type = 'VRG'
                fancy = None
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))

    parallel_async(func=generate_graphs, args=args, num_workers=10)
    # generate_graphs(grammar: Union[VRG, NCE, AttributedVRG], num_graphs: int, grammar_type: str, outdir: str = 'dumps',
    #                 mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None, fancy = None,
    #                 inp_deg_ast: float = None, inp_attr_ast: float = None)

    return


def batch_synthetic_generator_runner_attrs():
    # frac = np.linspace(0, 1, 21, endpoint=True) * 100
    frac = np.linspace(0, 100, 11, endpoint=True, dtype=int)  # change it to increments of 10 for now
    names = [f'toy-comm-{f}' for f in frac]
    # names = ['karate', 'football', 'polbooks', 'eucore', 'flights', 'chess', 'polblogs']
    num_graphs = 5
    outdir = '/data/ssikdar/attributed-vrg/dumps'
    use_pickle = True
    save_snapshots = False
    shuffle = 'attrs'

    args = []
    # input_graph, attr_name = nx.read_gexf(f'./input/shuffled/attrs/toy-comm-0.gexf', node_type=int), 'block'
    attr_name = 'block'
    for f in frac:
        g = nx.read_gexf(f'./input/shuffled/attrs/toy-comm-{f}.gexf', node_type=int)
        mix_dict = get_mixing_dict(g, attr_name=attr_name)
        inp_deg_ast = nx.degree_assortativity_coefficient(g)
        inp_attr_ast = nx.attribute_assortativity_coefficient(g, attr_name)
        name = f'toy-comm-attrs-{f}'

        for grammar_filename in glob(f'{outdir}/grammars/toy-comm-{f}-attrs/*'):
            grammar = load_pickle(grammar_filename)
            if isinstance(grammar, AttributedVRG):
                grammar_type = 'AVRG'
                fancy = True
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))
                grammar_type = 'AVRG-greedy'
                for alpha in (0, 0.5, 1):
                    args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                                 inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha))
            else:
                assert isinstance(grammar, VRG)
                grammar_type = 'VRG'
                fancy = None
                args.append((name, grammar, num_graphs, grammar_type, outdir, mix_dict, attr_name, fancy,
                             inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots))

    parallel_async(func=generate_graphs, args=args, num_workers=13)
    # generate_graphs(grammar: Union[VRG, NCE, AttributedVRG], num_graphs: int, grammar_type: str, outdir: str = 'dumps',
    #                 mixing_dict: Union[None, Dict] = None, attr_name: Union[str, None] = None, fancy = None,
    #                 inp_deg_ast: float = None, inp_attr_ast: float = None)

    return


def batched_graphs_clusters(basedir, name, clusterings, num_workers=5):
    input_graphs = read_batched_graphs(basedir=basedir, name=name)
    use_pickle = True
    args = []

    for i, g in enumerate(input_graphs):
        g.name = f'{name}-{i}'
        for clustering in clusterings:
            filename = join(basedir, 'output', 'trees', name, f'{clustering}_{i}.pkl')
            args.append((g, join(basedir, 'output', 'trees'), clustering, use_pickle, filename))

    parallel_async(func=get_clustering, args=args, num_workers=num_workers)
    # get_clustering(g: nx.Graph, outdir: str, clustering: str, use_pickle: bool, filename='') -> Any:
    return


def batched_graphs_grammars(basedir, name, clusterings):
    input_graphs = read_batched_graphs(basedir=basedir, name=name)
    attr_name = 'value'
    grammar_types = ['AVRG']  # ['VRG', 'AVRG']
    extract_types = ['mu_random']  #, 'mu_level', 'all_tnodes']
    mus = [5]
    use_cluster_pickle = True
    use_grammar_pickle = True
    count = 1

    args = []
    for i, input_graph in enumerate(input_graphs):
        for clustering in clusterings:
            list_of_list_clusters = load_pickle(join(basedir, 'output', 'trees', name, f'{clustering}_{i}.pkl'))
            for grammar_type in grammar_types:
                for extract_type in extract_types:
                    extract = extract_type.replace('_', '-')
                    for mu in mus:
                        grammar_filename = f'{basedir}/output/grammars/{name}/{grammar_type}_{extract}_{clustering}_{mu}_{i}.pkl'

                        arg = (name, grammar_type, extract_type, clustering, mu, input_graph, True,
                               True, attr_name, basedir, 1, grammar_filename, True, list_of_list_clusters)
                        args.append(arg)
                        if extract_type == 'all_tnodes':  # here mu is not important for all_tnodes
                            break

    # print(args[: 3])

    try:
        parallel_async(func=get_grammars, args=args, num_workers=5)
    except Exception as e:
        print(e)
    return
    # get_grammars(name: str,  grammar_type: str, extract_type: str, clustering: str, mu: int, input_graph: nx.Graph,
    #              use_grammar_pickle: bool, use_cluster_pickle: bool, attr_name: str, outdir: str, count: int = 1,
    #              grammar_filename: str = '', write_pickle: bool = True, list_of_list_clusters=None) -> List[Union[VRG, NCE]]:


def batched_graphs_generator(basedir, clusterings, name, mus=None):
    # num_graphs = 5 if 'polblogs' in name else 10
    num_graphs = 10
    use_pickle = True
    save_snapshots = False
    attr_name = 'value'
    mus = [5]
    alpha = None
    input_graphs = read_batched_graphs(basedir=basedir, name=name)
    extract_types = ['mu_random']

    args = []
    for i, input_graph in enumerate(input_graphs):
        mix_dict = get_mixing_dict(input_graph, attr_name=attr_name)
        inp_deg_ast = nx.degree_assortativity_coefficient(input_graph)
        inp_attr_ast = nx.attribute_assortativity_coefficient(input_graph, attr_name)

        for grammar_filename in glob(f'{basedir}/output/grammars/{name}/*_{i}.pkl'):
            grammar = load_pickle(grammar_filename)
            if grammar.mu not in mus or grammar.clustering not in clusterings or grammar.extract_type not in extract_types:
                continue

            extract_type = grammar.extract_type.replace('_', '-')
            if isinstance(grammar, AttributedVRG):
                for gen_type, fancy in zip(('AVRG-regular', 'AVRG-fancy'), (False, True)):
                    graphs_filename = f'{basedir}/output/graphs/{name}/{gen_type}_{extract_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}_{i}.pkl'
                    args.append((name, grammar, num_graphs, extract_type, gen_type, basedir, graphs_filename, mix_dict,
                                 attr_name, fancy, inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots, alpha))

                for alpha, gen_type in zip((0, 0.5, 1), ('AVRG-greedy-attr', 'AVRG-greedy-50', 'AVRG-greedy-deg')):
                    graphs_filename = f'{basedir}/output/graphs/{name}/{gen_type}_{extract_type}_{grammar.clustering}_{grammar.mu}_{num_graphs}_{i}.pkl'
                    args.append((name, grammar, num_graphs, extract_type, gen_type, basedir, graphs_filename,
                                 mix_dict, attr_name, fancy, inp_deg_ast, inp_attr_ast, use_pickle, save_snapshots,
                                 alpha))

    # random.shuffle(args)
    parallel_async(func=generate_graphs, args=args, num_workers=8)
    return


@timer
def motif_counter_runner(name, nx_graph, basedir, overwrite, motif_fname=None):
    ig_g = nx_to_igraph(nx_graph)
    mc = MotifCounter(name=name, input_graph=ig_g, basedir=basedir)

    start = time.perf_counter()
    mc.count(ks=[3, 4], overwrite=overwrite, fname=motif_fname)
    # mc.plot_motifs()
    end = time.perf_counter()
    print(f'Counting motifs for {name!r} took {end - start:.2g} sec')
    return


def batch_motif_counter(name, model, basedir, overwrite=False, graphs=None, motif_filename=None):
    # dont overwrite by default
    args = []
    # motif_counter_runner(name, nx_graph, basedir, overwrite, motif_fname=None)
    for i, graph in enumerate(graphs):
        # if motif_filename is None:
        if model == 'original':
            model_ = model
        else:
            model_ = f'{model}_{i}'
        motif_filename = join(basedir, 'output/motifs/', name, f'{model_}.pkl')
        args.append((name, graph, basedir, overwrite, motif_filename))
    try:
        parallel_async(func=motif_counter_runner, args=args, num_workers=5)
    except Exception as e:
        logging.error(e)
    return


def get_best_graphs():
    basedir = '/data/ssikdar/Attributed-VRG'
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'airports', 'polblogs',
             'film', 'chameleon', 'squirrel']
    # names = ['polbooks']
    clusterings = ['leiden']
    extract_types = ['mu-random']
    mus = [5]

    for name in names:
        make_dirs(outdir=join(basedir, 'output'), name=name)

    batch_generator_runner(basedir=basedir, names=names, clusterings=clusterings, mus=mus, extract_types=extract_types,
                           shuffle=False, num_workers=8)
    exit(1)


def get_best_grammars():
    basedir = '/data/ssikdar/Attributed-VRG'
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'airports', 'polblogs', 'film', 'chameleon', 'squirrel']
    # names = ['polbooks']
    clusterings = ['leiden']
    extract_types = ['mu-random']
    mus = [5]

    for name in names:
        make_dirs(outdir=join(basedir, 'output'), name=name)
    batch_grammar_runner(names=names, clusterings=clusterings, mus=mus, extract_types=extract_types,
                         outdir=basedir, shuffle=False, num_workers=8)
    return


def old_main():
    basedir = '/data/ssikdar/Attributed-VRG'

    # names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'airports',
    #          'polblogs', 'film', 'chameleon', 'squirrel']
    #
    # clusterings = ['cond', 'leiden', 'louvain', 'spectral', 'infomap', 'labelprop', 'random'][: 3]
    #
    # for name in names:
    #     make_dirs(outdir=join(basedir, 'output'), name=name)

    # batch_motif_counter(names=names, basedir=basedir)
    # batch_cluster_runner(names=names, clusterings=clusterings, outdir=basedir)
    # batch_grammar_runner(names=names, clusterings=clusterings, outdir=basedir, shuffle=False, num_workers=8)
    # batch_generator_runner(basedir=basedir, names=names, clusterings=clusterings, shuffle=False, num_workers=5)
    # return

    # clusterings = ['cond', 'spectral', 'leiden', 'louvain', 'random']

    # clusterings = ['cond', 'louvain', 'leiden', 'spectral']
    # name = 'polbooks'
    # for kind in ('deg', 'attr')[1: ]:
    #     name_ = f'{name}-{kind}'
    #     # batched_graphs_clusters(outdir, name=name_, clusterings=clusterings)
    #     # batched_graphs_grammars(outdir=outdir, name=name_, clusterings=clusterings)
    #     batched_graphs_generator(outdir=outdir, name=name_, clusterings=clusterings)
    #
    # # batch_synthetic_generator_runner_attrs()
    # # batch_synthetic_generator_runner()

    # names = ['lang-bip']
    # clusterings = ['cond', 'leiden', 'louvain', 'spectral']
    # batch_cluster_shuffler_runner(names=names, clusterings=clusterings)
    # batch_cluster_runner(names=names, clusterings=clusterings, outdir=outdir)

    # batch_generator_runner(names=names, clusterings=clusterings, outdir=outdir)


def cabam():
    basedir = '/data/ssikdar/Attributed-VRG/'
    clusterings = ['leiden']
    # batched_graphs_clusters(clusterings=clusterings, basedir=basedir, name='cabam')
    # batched_graphs_grammars(clusterings=clusterings, basedir=basedir, name='cabam')
    batched_graphs_generator(clusterings=clusterings, basedir=basedir, name='cabam')
    return


def colored_motifs():
    basedir = '/data/ssikdar/Attributed-VRG/'
    names = ['polbooks', 'football', 'wisconsin', 'texas', 'cornell', 'polblogs']
    # names = ['citeseer', 'cora', 'airports']
    models = ['AVRG', 'CL', 'AGM', 'DC-SBM', 'CELL', 'NetGAN', 'original']

    for name in names:
        for model in models:
            print(f'Running {name!r} {model!r}')
            if model == 'AVRG':
                model_ = 'AVRG-fancy_mu-random_leiden_5'
            else:
                model_ = model

            graphs_filename = join(basedir, 'output/graphs/', name, f'{model_}_10.pkl')
            if model == 'original':
                graphs = [nx.read_gml(join(basedir, 'input', f'{name}.gml'))]
            else:
                graphs = load_pickle(graphs_filename)
            if graphs is None:
                continue
            # batch_motif_counter(name, model, basedir, overwrite=False, graphs=None, motif_filename=None):
            batch_motif_counter(name=name, model=model, basedir=basedir, graphs=graphs, overwrite=False)

    return


if __name__ == '__main__':
    # get_best_grammars()
    # get_best_graphs()
    cabam()
    # colored_motifs()
