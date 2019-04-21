from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score,accuracy_score

from optimizer import OptimizerAE, OptimizerVAE
# from input_data import load_data
from util import *
from model import GCNModelAE, GCNModelVAE
from alignments import get_align

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for ae.')
flags.DEFINE_float('learning_rate_g', 0.001, 'Initial learning rate for ae.')
flags.DEFINE_float('learning_rate_d', 0.001, 'Initial learning rate for ae.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in Discriminator.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('d', 1, 'num for d to update.')
flags.DEFINE_integer('g', 1, 'num for g to update.')
flags.DEFINE_integer('ae', 1, 'num to pre-train ae.')
flags.DEFINE_integer('feature', 1, '(1) use feature (0) not use feature.')
model_str = 'gcn_ae' # or gcn_vae


S1, S2, A1, A2, y_train1, y_train2, y_test1, y_test2 = load_data('dataset/Douban.mat',0.5)
S1_p = preprocess_adj(S1)
S2_p = preprocess_adj(S2)
pos_weight1 = float(S1.shape[0] * S1.shape[0] - S1.sum()) / S1.sum()
norm1 = S1.shape[0] * S1.shape[0] / float((S1.shape[0] * S1.shape[0] - S1.sum()) * 2)
pos_weight2 = float(S2.shape[0] * S2.shape[0] - S2.sum()) / S2.sum()
norm2 = S2.shape[0] * S2.shape[0] / float((S2.shape[0] * S2.shape[0] - S2.sum()) * 2)
S1_ori = sparse_to_tuple(S1+sp.eye(S1.shape[0]))
S2_ori = sparse_to_tuple(S2+sp.eye(S2.shape[0]))
if FLAGS.feature==0:
    A1 = sp.identity(A1.shape[0])
    A2 = sp.identity(A2.shape[0])
A1 = sparse_to_tuple(A1)
A2 = sparse_to_tuple(A2)


# Define placeholders
placeholders = [{
    # 'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32,shape=tf.constant(A1[2], dtype=tf.int64)),
}, {
    # 'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32,shape=tf.constant(A2[2], dtype=tf.int64)),
}]
dropout = tf.placeholder(tf.float32)
num_nodes = [S1_ori[2][0],S2_ori[2][0]]
num_features = [A1[2][1],A2[2][1]]
features_nonzero = [A1[1].shape[0],A2[1].shape[0]]

# Create model
#map_model = Discriminator()
# model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features = num_features, features_nonzero = features_nonzero,dropout=dropout,flag=True)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features = num_features,num_nodes=num_nodes, features_nonzero = features_nonzero,dropout=dropout)


pos_weight1 = float(S1.shape[0] * S1.shape[0] - S1.sum()) / S1.sum()
norm1 = S1.shape[0] * S1.shape[0] / float((S1.shape[0] * S1.shape[0] - S1.sum()) * 2)
pos_weight2 = float(S2.shape[0] * S2.shape[0] - S2.sum()) / S2.sum()
norm2 = S2.shape[0] * S2.shape[0] / float((S2.shape[0] * S2.shape[0] - S2.sum()) * 2)

pos_weight = [pos_weight1,pos_weight2]
norm = [norm1,norm2]

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=[model.reconstructions1,model.reconstructions2],
                          # labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                        labels=[tf.reshape(tf.sparse_tensor_to_dense(placeholders[0]['adj_orig'],
                                                                      validate_indices=False),[-1]),\
                                tf.reshape(tf.sparse_tensor_to_dense(placeholders[1]['adj_orig'],
                                                                      validate_indices=False),[-1])],
                        pos_weight=pos_weight,
                        norm=norm,
                        fake_emd=model.output,
                        model=model,
                        real_emd=model.z_mean2,
                        y1=y_train1,y2=y_train2,
                        size1=S1.shape[0],size2=S2.shape[0])
    # elif model_str == 'gcn_vae':
    #     opt = OptimizerVAE(preds=[model.reconstructions1, model.reconstructions2],
    #                        # labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
    #                        labels=[tf.reshape(tf.sparse_tensor_to_dense(placeholders[0]['adj_orig'],
    #                                                                   validate_indices=False),[-1]),\
    #                             tf.reshape(tf.sparse_tensor_to_dense(placeholders[1]['adj_orig'],
    #                                                                   validate_indices=False),[-1])],
    #                        pos_weight=pos_weight,
    #                        norm=norm,
    #                        fake_emd=model.output,
    #                        model_d=map_model,
    #                        model_g=model,
    #                        num_nodes=num_nodes,
    #                        real_emd=model.z_mean2,y1=y_train1,y2=y_train2)

# Initialize session
merge = tf.summary.merge_all()
sess = tf.Session()
train_write = tf.summary.FileWriter('./model_log',sess.graph)
sess.run(tf.global_variables_initializer())
feed_dict_val = construct_feed_dict([A1,A2], [S1_p,S2_p],[S1_ori,S2_ori], placeholders)
feed_dict_val.update({dropout:FLAGS.dropout})
best_acc = 0
# print(tf.trainable_variables())
# pre training autoencoders
for i in range(30):
    # test = sess.run([model.reconstructions1,model.reconstructions2], feed_dict=feed_dict_val)
    _, ae_loss,avg1,avg2 = sess.run([opt.opt_op, opt.cost,opt.accuracy1,opt.accuracy2], feed_dict=feed_dict_val)
    # test = sess.run([model.z_mean1, model.z_mean2], feed_dict=feed_dict_val)
    print("Epoch:", '%04d' % (i + 1), "loss=", "{:.5f}".format(ae_loss))

# Train model
for epoch in range(FLAGS.epochs+1):
   # if epoch>250:
    #    feed_dict_val.update({lr:0.0001})
    for i in range(FLAGS.ae):
        _, ae_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict_val)
#print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.5f}".format(ae_loss))
# for d in range(FLAGS.d):
#     _,d_loss = sess.run([opt.discriminator_optimizer,opt.dc_loss],feed_dict=feed_dict_val)
#
# for g in range(FLAGS.g):
#     _,g_loss = sess.run([opt.generator_optimizer,opt.generator_loss],feed_dict=feed_dict_val)
# print("Epoch:", '%04d' % (epoch + 1), "loss_d=", "{:.5f}".format(d_loss),"loss_g=",'{:.5f}'.format(g_loss))

### test mapping function
    feed_dict_val.update({dropout: FLAGS.dropout})
    for d in range(FLAGS.d):
        _ = sess.run(opt.clip_D)
        _,Dis = sess.run([opt.op_d,opt.D],feed_dict=feed_dict_val)
    print("Epoch:", '%04d' % (epoch + 1), "loss_D=", '{:.5f}'.format(Dis))
    _, Gen = sess.run([opt.op_g, opt.G], feed_dict=feed_dict_val)
    print("Epoch:", '%04d' % (epoch + 1), "loss_G=", '{:.5f}'.format(Gen))
    _, mapping = sess.run([opt.map_op, opt.mapping], feed_dict=feed_dict_val)
    print("mapping loss",'{:.5f}'.format(mapping))
    # summary = sess.run(merge,feed_dict=feed_dict_val)
    # train_write.add_summary(summary,epoch)
    if epoch%10==0:
        feed_dict_val.update({dropout: 0.})
        emb1,emb2 = sess.run([model.output,model.z_mean2],feed_dict=feed_dict_val)
        pred = get_align(emb1,emb2,num_top=50)
        train_acc = 0
        for i in range(len(y_train2)):
            if y_train1[i] in np.argsort(pred[y_train2[i]])[-10:]:
                train_acc+=1
        print('training acc','{:.5f}'.format(train_acc/len(y_train2)))
        test_acc = 0
        for i in range(len(y_test2)):
            if y_test1[i] in np.argsort(pred[y_test2[i]])[-10:]:
                test_acc+=1
        print('testing acc','{:.5f}'.format(test_acc/len(y_test2)))
        print('all_acc','{:.5f}'.format((test_acc+train_acc)/(len(y_train2)+len(y_test2))))
        if best_acc<test_acc:
            best_acc = test_acc
            itr = epoch
            sim = pred
# with open('models/f1'+data_str+'_'+str(ratio),'wb') as f:
# #    pickle.dump(best_acc,f)
# #    pickle.dump(itr,f)
#     pickle.dump(sim,f)
#     pickle.dump(y_test1,f)
#     pickle.dump(y_test2,f)
# f.close()
print("best_acc:",best_acc/len(y_test2))
print('best_itr:',itr)

# eval the performance of model
# print("not implement the evaluation!")