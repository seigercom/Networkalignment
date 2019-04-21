import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy.io as sio
import tensorflow as tf
import time
from scipy.sparse import csr_matrix

# def parse_index_file(filename):
#     """Parse index file."""
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index


# def sample_mask(idx, l):
#     """Create mask."""
#     mask = np.zeros(l)
#     mask[idx] = 1
#     return np.array(mask, dtype=np.bool)


def load_data(dataset_str,ratio):
    """
    Loads input data from gcn/data directory


    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # douban, train_flickr-lastfm, train_flickr-myspace should minus 1
    if dataset_str=='dataset/Douban.mat':
        blog = sio.loadmat(dataset_str)
        S1 = sp.coo_matrix(blog['online'])
        S2 = sp.coo_matrix(blog['offline'])
        A1 = sp.coo_matrix(blog['online_node_label'])
        A2 = sp.coo_matrix(blog['offline_node_label'])
        groundTruth = blog['ground_truth']
        # groundTruth = groundTruth-np.ones(groundTruth.shape)
        # np.random.shuffle(groundTruth)
        tmp = int(ratio * len(groundTruth))
        train, test = groundTruth[:tmp], groundTruth[tmp:]
        train_idx1, train_idx2 = train[:,0]-1,train[:,1]-1
        train_idx1 = train_idx1.astype(np.int32)
        train_idx2 = train_idx2.astype(np.int32)
        test_idx1, test_idx2 = test[:,0]-1, test[:,1]-1
    elif 'train_' in dataset_str:
        blog = sio.loadmat(dataset_str)
        S1 = sp.coo_matrix(blog['adj1'])
        S2 = sp.coo_matrix(blog['adj2'])
        A1 = sp.coo_matrix(blog['atr1'])
        A2 = sp.coo_matrix(blog['atr2'])
        groundTruth = blog['gndtruth']
        np.random.shuffle(groundTruth)
        tmp = int(ratio * len(groundTruth))
        train, test = groundTruth[:tmp], groundTruth[tmp:]
        train_idx1, train_idx2 = train[:, 0]-1 , train[:, 1]-1
        train_idx1 = train_idx1.astype(np.int32)
        train_idx2 = train_idx2.astype(np.int32)
        test_idx1, test_idx2 = test[:, 0]-1 , test[:, 1]-1
    else:
        blog = sio.loadmat(dataset_str)
        S1 = sp.coo_matrix(blog['adj1'])
        S2 = sp.coo_matrix(blog['adj2'])
        A1 = sp.coo_matrix(blog['atr1'])
        A2 = sp.coo_matrix(blog['atr2'])
        groundTruth = blog['groundTruth']
        np.random.shuffle(groundTruth)
        tmp = int(ratio * len(groundTruth))
        train, test = groundTruth[:tmp], groundTruth[tmp:]
        train_idx1, train_idx2 = train[:, 0], train[:, 1]
        train_idx1 = train_idx1.astype(np.int32)
        train_idx2 = train_idx2.astype(np.int32)
        test_idx1, test_idx2 = test[:, 0], test[:, 1]


    return S1, S2, A1, A2, train_idx2, train_idx1, test_idx2, test_idx1  #coo shape. train/test is the label of alignment of 2 networks


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support,adj_ori, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    # feed_dict.update({placeholders[0]['labels']: labels[0]})
    # feed_dict.update({keep_prob:dropout})
    feed_dict.update({placeholders[0]['features']: features[0]})
    feed_dict.update({placeholders[0]['adj']: support[0]})
    feed_dict.update({placeholders[0]['adj_orig']: adj_ori[0]})
    # feed_dict.update({placeholders[0]['num_features_nonzero']: features[0][1].shape[0]})

    # feed_dict.update({placeholders[1]['labels']: labels[1]})

    feed_dict.update({placeholders[1]['features']: features[1]})
    feed_dict.update({placeholders[1]['adj']: support[1]})
    feed_dict.update({placeholders[1]['adj_orig']: adj_ori[1]})
    # feed_dict.update({placeholders[1]['num_features_nonzero']: features[1][1].shape[0]})
    return feed_dict







