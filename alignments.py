import numpy as np
import scipy.io as sio
import sklearn.metrics.pairwise
from scipy.sparse import csc_matrix, coo_matrix,csr_matrix
from sklearn.neighbors import KDTree
import scipy.sparse as sp
from scipy.spatial.distance import cosine

def get_embedding_similarities(embed, embed2 = None, sim_measure = "euclidean", num_top = None):
    n_nodes, dim = embed.shape
    if embed2 is None:
        embed2 = embed

    if num_top is not None: #KD tree with only top similarities computed
        kd_sim = kd_align(embed, embed2, distance_metric = sim_measure, num_top = num_top)
        return kd_sim

  #All pairwise distance computation
    if sim_measure == "cosine":
        similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(embed, embed2)
    else:
        similarity_matrix = sklearn.metrics.pairwise.euclidean_distances(embed, embed2)
        similarity_matrix = np.exp(-similarity_matrix)

    return similarity_matrix


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=1):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    # data = dist.flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix

def greed_match(sim):
    m = sim.shape[0]
    n = sim.shape[1]
    flated = sim.ravel()
    usedRows = np.zeros(m)
    usedCols = np.zeros(n)
    # ind = sorted(range(len(flated)),reverse=True,key=lambda k: flated[k])
    ind = np.argsort(flated)[::-1]
    minSize = min(m,n)

    row =np.zeros(minSize)
    col = np.zeros(minSize)
    i = 0
    matched = 0
    while matched<minSize:
        ipos = ind[i]
        jc = int(np.floor(ipos/n))
        ic = int(ipos-jc*n)
        if usedCols[ic]!=1 and usedRows[jc]!=1:
            row[matched] = jc
            col[matched] = ic
            usedRows[jc] = 1
            usedCols[ic] = 1
            matched += 1
        i+=1
    data = np.ones(minSize)
    return csr_matrix((data, (row, col)), shape=(m, n))

def get_align(emb1,emb2,sim_measure = "euclidean", num_top = None):
    sim = get_embedding_similarities(emb1,emb2,sim_measure=sim_measure,num_top=num_top).toarray()
    return greed_match(sim)
#
# X1= np.array([[.1,.2,.3],[.4,.5,.6]])
# X2 =np.array([[.41,.51,.61],[.11,.21,.31],[1,2,3]])
# sim = sio.loadmat('sim.mat')['S']
# label = sio.loadmat('sim.mat')['ground_truth']
# # # test = (label[0][1],label[0][0])
# t = greed_match(sim)
# train_acc = 0
# for i in range(len(label)):
#     if t[label[i][1]-1,label[i][0]-1]==1:
#         train_acc+=1
# print('training acc','{:.5f}'.format(train_acc/len(label)))
# print('test')