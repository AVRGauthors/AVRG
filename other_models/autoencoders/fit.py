from __future__ import division
from __future__ import print_function
from other_models.autoencoders.evaluation import get_roc_score, clustering_latent_space, get_prob_mat_from_emb
from other_models.autoencoders.input_data import load_data, load_label
from other_models.autoencoders.kcore import compute_kcore, expand_embedding
from other_models.autoencoders.model import *
from other_models.autoencoders.optimizer import OptimizerAE, OptimizerVAE
from other_models.autoencoders.preprocessing import *
import networkx as nx
import numpy as np
from collections import namedtuple
import os
import scipy.sparse as sp
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = namedtuple('FLAGS', ['dataset', 'task', 'model', 'dropout', 'epochs', 'features', 'learning_rate', 
    'hidden', 'dimension', 'nb_run', 'prop_val', 'prop_test', 'validation', 'verbose', 'kcore', 'k', 'nb_iterations'])

FLAGS = flags('custom', 'link_prediction', 'gcn_ae', 0., 200, False, 0.01, 32, 16, 1, 5., 10., False, True, False, 2, 10)


def fit_model(g, model_name):
    # Lists to collect average results
    mean_roc = []
    mean_ap = []
    mean_time = []
    # Load graph dataset
    adj_init = nx.adjacency_matrix(g)
    features_init = sp.eye(g.order(), g.size())
    tf.disable_eager_execution()
    # tf.compat.v1.disable_eager_execution()
    print(f"Loading data... {g.name} n: {g.order()}, m: {g.size()}")

    # The entire training+test process is repeated FLAGS.nb_run times
    for i in range(FLAGS.nb_run):
        if FLAGS.task == 'link_prediction' :
            print("Masking test edges...")
            # Edge Masking for Link Prediction: compute Train/Validation/Test set
            while True:
                try:
                    adj, val_edges, val_edges_false, test_edges, test_edges_false = \
                        mask_test_edges(adj_init, FLAGS.prop_test, FLAGS.prop_val)
                except Exception:
                    continue
                else:
                    break
        else:
            raise ValueError('Undefined task!')

        # Start computation of running times
        t_start = time.time()

        # Preprocessing and initialization
        print("Preprocessing and Initializing...")
        # Compute number of nodes
        num_nodes = adj.shape[0]
        # If features are not used, replace feature matrix by identity matrix
        if not FLAGS.features:
            features = sp.identity(adj.shape[0])
        # Preprocessing on node features
        features = sparse_to_tuple(features)
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        # Define placeholders
        placeholders = {
            'features': tf.compat.v1.sparse_placeholder(tf.float32),
            'adj': tf.compat.v1.sparse_placeholder(tf.float32),
            'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape = ())
        }

        # Create model
        model = None
        if model_name == 'gcn_ae':
            # Standard Graph Autoencoder
            model = GCNModelAE(placeholders, num_features, features_nonzero)
        elif model_name == 'gcn_vae':
            # Standard Graph Variational Autoencoder
            model = GCNModelVAE(placeholders, num_features, num_nodes,
                                features_nonzero)
        elif model_name == 'linear_ae':
            # Linear Graph Autoencoder
            model = LinearModelAE(placeholders, num_features, features_nonzero)
        elif model_name == 'linear_vae':
            # Linear Graph Variational Autoencoder
            model = LinearModelVAE(placeholders, num_features, num_nodes,
                                   features_nonzero)
        elif model_name == 'deep_gcn_ae':
            # Deep (3-layer GCN) Graph Autoencoder
            model = DeepGCNModelAE(placeholders, num_features, features_nonzero)
        elif model_name == 'deep_gcn_vae':
            # Deep (3-layer GCN) Graph Variational Autoencoder
            model = DeepGCNModelVAE(placeholders, num_features, num_nodes,
                                    features_nonzero)
        else:
            raise ValueError('Undefined model!')

        # Optimizer
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0]
                                                    - adj.sum()) * 2)
        with tf.name_scope('optimizer'):
            # Optimizer for Non-Variational Autoencoders
            if model_name in ('gcn_ae', 'linear_ae', 'deep_gcn_ae'):
                opt = OptimizerAE(preds = model.reconstructions,
                                  labels = tf.reshape(tf.compat.v1.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                validate_indices = False), [-1]),
                                  pos_weight = pos_weight,
                                  norm = norm)
            # Optimizer for Variational Autoencoders
            elif model_name in ('gcn_vae', 'linear_vae', 'deep_gcn_vae'):
                opt = OptimizerVAE(preds = model.reconstructions,
                                   labels = tf.reshape(tf.compat.v1.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                 validate_indices = False), [-1]),
                                   model = model,
                                   num_nodes = num_nodes,
                                   pos_weight = pos_weight,
                                   norm = norm)

        # Normalization and preprocessing on adjacency matrix
        adj_norm = preprocess_graph(adj)
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

        # Initialize TF session
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # Model training
        print(f"Training {model_name}...")

        t = time.time()
        print_every = 50
        for epoch in range(FLAGS.epochs):
            # Flag to compute running time for each epoch
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Weights update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                            feed_dict = feed_dict)
            # Compute average loss
            avg_cost = outs[1]
            if epoch > 0 and epoch % print_every == 0 and FLAGS.verbose:
                # Display epoch information
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(avg_cost),
                        "time/epoch: {:.5f}s".format((time.time() - t) / print_every))
                t = time.time()  # reset the clock
                if not FLAGS.kcore and FLAGS.validation and FLAGS.task == 'link_prediction':
                    feed_dict.update({placeholders['dropout']: 0})
                    emb = sess.run(model.z_mean, feed_dict = feed_dict)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    val_roc, val_ap = get_roc_score(val_edges, val_edges_false, emb)
                    print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))
        # Flag to compute Graph AE/VAE training time
        t_model = time.time()

        # Compute embedding
        # Get embedding from model
        emb = sess.run(model.z_mean, feed_dict = feed_dict)
        mean_time.append(time.time() - t_start)

        # Test model
        print("Testing model...")
        # Link Prediction: classification edges/non-edges
        if FLAGS.task == 'link_prediction':
            # Get ROC and AP scores
            roc_score, ap_score = get_roc_score(test_edges, test_edges_false, emb)
            # Report scores
            mean_roc.append(roc_score)
            mean_ap.append(ap_score)

    sess.close()  # close the session and free up resouces
    ### SS: compute final graph 
    prob_mat, thresh_mat = get_prob_mat_from_emb(emb)
    return prob_mat, thresh_mat


if __name__ == '__main__':
    g = nx.karate_club_graph()
    g.name = 'karate'
    model_name = 'gcn_ae'
    prob_mat = fit_model(g, model_name)
    gen_g = nx.from_numpy_matrix(prob_mat, create_using=nx.Graph())
    print(f'{g.name} orig: n={g.order()} m={g.size()} | gen: n={gen_g.order()} m={gen_g.size()}')

