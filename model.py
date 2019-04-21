from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out

class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero,dropout,flag=False, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs1 = placeholders[0]['features']
        self.input_dim1 = num_features[0]
        self.features_nonzero1 = features_nonzero[0]
        self.adj1 = placeholders[0]['adj']

        self.inputs2 = placeholders[1]['features']
        self.input_dim2 = num_features[1]
        self.features_nonzero2 = features_nonzero[1]
        self.adj2 = placeholders[1]['adj']
        self.flag = flag
        self.dropout = dropout
        self.build()

    def _build(self):
        with tf.name_scope('Autoencoder'):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim1,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj1,
                                                  features_nonzero=self.features_nonzero1,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  name='e_h1',
                                                  logging=self.logging)(self.inputs1)

            self.embeddings1 = GraphConvolution(input_dim=FLAGS.hidden1,
                                               output_dim=FLAGS.hidden2,
                                               adj=self.adj1,
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                                name='e_1',
                                               logging=self.logging)(self.hidden1)

            self.z_mean1 = self.embeddings1

            self.reconstructions1 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                          act=lambda x: x,
                                          logging=self.logging)(self.embeddings1)

            self.hidden2 = GraphConvolutionSparse(input_dim=self.input_dim2,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj2,
                                                  features_nonzero=self.features_nonzero2,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  name='e_h2',
                                                  logging=self.logging)(self.inputs2)

            self.embeddings2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden2,
                                                adj=self.adj2,
                                                act=lambda x: x,
                                                dropout=self.dropout,
                                                name='e_2',
                                                logging=self.logging)(self.hidden2)

            self.z_mean2 = self.embeddings2

            self.reconstructions2 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                        act=lambda x: x,
                                                        logging=self.logging)(self.embeddings2)

        # MLP mapping to network2
        # non-linear
        if self.flag:
            dc_den1 = tf.nn.relu(dense(self.z_mean1, FLAGS.hidden2, FLAGS.hidden3, name='g_den1'))
            self.output = dense(dc_den1, FLAGS.hidden3, FLAGS.hidden2, name='g_output')
        # linear
        if not self.flag:
            self.output = dense(self.z_mean1,FLAGS.hidden2,FLAGS.hidden2,name='g_den1')

    def discriminator(self,input,label=None):
        hid1 = tf.nn.relu(dense(input,FLAGS.hidden2,16,name='d_1'))
        logits = dense(hid1,16,1,name='d_2')
        # GAN
        #dis = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=label)
        # WGAN
        dis = logits
        return dis


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero,dropout,flag=False, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs1 = placeholders[0]['features']
        self.input_dim1 = num_features[0]
        self.features_nonzero1 = features_nonzero[0]
        self.adj1 = placeholders[0]['adj']
        self.n_samples1 = num_nodes[0]

        self.inputs2 = placeholders[1]['features']
        self.input_dim2 = num_features[1]
        self.features_nonzero2 = features_nonzero[1]
        self.adj2 = placeholders[1]['adj']
        self.n_samples2 = num_nodes[1]
        self.flag = flag

        self.dropout = dropout
        self.build()

    def _build(self):
        with tf.name_scope('Autoencoder'):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim1,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj1,
                                                  features_nonzero=self.features_nonzero1,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  name='e_h1',
                                                  logging=self.logging)(self.inputs1)

            self.z_mean1 = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj1,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                            name='e_mean1',
                                           logging=self.logging)(self.hidden1)

            self.z_log_std1 = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj1,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                               name='e_log_std1',
                                              logging=self.logging)(self.hidden1)

            self.z1 = self.z_mean1 + tf.random_normal([self.n_samples1, FLAGS.hidden2]) * tf.exp(self.z_log_std1)

            self.reconstructions1 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                          act=lambda x: x,
                                          logging=self.logging)(self.z1)

            self.hidden2 = GraphConvolutionSparse(input_dim=self.input_dim2,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj2,
                                                  features_nonzero=self.features_nonzero2,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  name='e_h2',
                                                  logging=self.logging)(self.inputs2)

            self.z_mean2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj2,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                            name='e_mean2',
                                           logging=self.logging)(self.hidden2)

            self.z_log_std2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj2,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                               name='e_std2',
                                              logging=self.logging)(self.hidden2)

            self.z2 = self.z_mean2 + tf.random_normal([self.n_samples2, FLAGS.hidden2]) * tf.exp(self.z_log_std2)

            self.reconstructions2 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                       act=lambda x: x,
                                                       logging=self.logging)(self.z2)

        # MLP mapping to network2
            if self.flag:
                dc_den1 = tf.nn.relu(tf.nn.dropout(dense(self.z_mean1, FLAGS.hidden2, FLAGS.hidden3, name='e_den1'),1-self.dropout))
                self.output = dense(dc_den1, FLAGS.hidden3, FLAGS.hidden2, name='e_output')
            if not self.flag:
                self.output = dense(self.z_mean1,FLAGS.hidden2,FLAGS.hidden2,name = 'e_den1')










# class Discriminator(Model):
#     def __init__(self, **kwargs):
#         super(Discriminator, self).__init__(**kwargs)
#         # self.build()
#         # self.act = tf.nn.relu
#
#     def construct(self,input):
#         # with tf.name_scope('Discriminator'):
#         self.hid1 = tf.nn.relu(dense(input,FLAGS.hidden2,FLAGS.hidden3,name='d_h1'))
#         self.hid2 = tf.nn.relu(dense(self.hid1, FLAGS.hidden3, FLAGS.hidden2, name='d_h2'))
#         self.hid3 = dense(self.hid2, FLAGS.hidden2, 1, name='d_h3')
#         return self.hid3



