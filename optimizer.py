import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm, real_emd,fake_emd,model,y1,y2,size1,size2):
        self.real = tf.nn.embedding_lookup(real_emd,y1)
        self.fake = tf.nn.embedding_lookup(fake_emd,y2)
        batch1 = size1
        batch2 = size2
        batch = self.real.get_shape()[0]
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd' in var.name]
        g_vars = [var for var in t_vars if 'g' in var.name]

        self.cost = norm[0] * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds[0], targets=labels[0], pos_weight=pos_weight[0])) + \
                    norm[1] * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds[1], targets=labels[1], pos_weight=pos_weight[1]))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer


        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.mapping = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.real-self.fake),axis=1)))#+self.cost
        self.map_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.mapping)


        self.correct_prediction1 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds[0]), 0.5), tf.int32),tf.cast(labels[0], tf.int32))
        self.correct_prediction2 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds[1]), 0.5), tf.int32),
                                           tf.cast(labels[1], tf.int32))
        self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))
        self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32))
        # GAN
        #self.G = tf.reduce_mean(model.discriminator(self.fake,tf.ones([batch,1]))) + self.mapping
        #self.D = tf.reduce_mean(model.discriminator(self.real,tf.ones([batch,1]))) + tf.reduce_mean(model.discriminator(self.fake,tf.zeros([batch,1])))
        # WGAN
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in d_vars]
        self.G = -tf.reduce_mean(model.discriminator(self.fake)) #+ self.mapping
        self.D = tf.reduce_mean(model.discriminator(self.real)) - tf.reduce_mean(
             model.discriminator(self.fake))

        self.op_g = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_g).minimize(self.G,var_list=g_vars)
        self.op_d = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_d).minimize(self.D,var_list=d_vars)

class OptimizerVAE(object):
    def __init__(self, preds, labels, model_g,model_d, num_nodes, pos_weight,norm, real_emd,fake_emd,y1,y2):
        self.real = tf.nn.embedding_lookup(model_d.construct(real_emd), y1)
        self.fake = tf.nn.embedding_lookup(model_d.construct(fake_emd), y2)


        self.cost =norm[0] * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds[0], targets=labels[0], pos_weight=pos_weight[0])) + \
                    norm[1] * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds[1], targets=labels[1], pos_weight=pos_weight[1]))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes[0]) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model_g.z_log_std1 - tf.square(model_g.z_mean1) -
                                                                   tf.square(tf.exp(model_g.z_log_std1)), 1)) + \
                  (0.5 / num_nodes[1]) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model_g.z_log_std2 - tf.square(model_g.z_mean2) -
                                                                   tf.square(tf.exp(model_g.z_log_std2)), 1))
        self.cost -= self.kl



        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.mapping = tf.reduce_mean(tf.sqrt(
            tf.reduce_sum(tf.square(tf.nn.embedding_lookup(real_emd, y1) - tf.nn.embedding_lookup(fake_emd, y2)),
                          axis=1))) + self.cost
        self.map_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.mapping)
        self.correct_prediction1 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds[0]), 0.5), tf.int32),
                                            tf.cast(labels[0], tf.int32))
        self.correct_prediction2 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds[1]), 0.5), tf.int32),
                                            tf.cast(labels[1], tf.int32))
        self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))
        self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32))