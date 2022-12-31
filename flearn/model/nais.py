# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import logging
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()
class NAIS:
    def __init__(self, num_items, args):
        self.pretrain = args.pretrain
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.weight_size = args.weight_size
        self.alpha = args.alpha
        self.beta = args.beta
        self.data_alpha = args.data_alpha
        self.verbose = args.verbose
        self.activation = args.activation
        self.algorithm = args.algorithm
        self.batch_choice = args.batch_choice
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.eta_bilinear = regs[2]
        self.train_loss = args.train_loss
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            print("initialized")

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])  # the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])  # the ground truth

    def _create_variables(self):
        with tf.variable_scope("embedding"):  # The embedding initialization is unknown now
            trainable_flag = (self.pretrain != 2)
            self.c1 = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='c1',
                dtype=tf.float32, trainable=trainable_flag)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32, trainable=trainable_flag)
            self.bias = tf.Variable(tf.zeros(self.num_items), name='bias', dtype=tf.float32, trainable=trainable_flag)

            # Variables for attention
            if self.algorithm == 0:
                self.W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0,
                                                         stddev=tf.sqrt(
                                                             tf.div(2.0, self.weight_size + self.embedding_size))),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:
                self.W = tf.Variable(tf.truncated_normal(shape=[2 * self.embedding_size, self.weight_size], mean=0.0,
                                                         stddev=tf.sqrt(tf.div(2.0, self.weight_size + (
                                                         2 * self.embedding_size)))), name='Weights_for_MLP',
                                     dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.weight_size + self.embedding_size))), name='Bias_for_MLP', dtype=tf.float32,
                                 trainable=True)
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

    def _attention_MLP(self, q_):
        with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            r = (self.algorithm + 1) * self.embedding_size

            MLP_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            if self.activation == 0:
                MLP_output = tf.nn.relu(MLP_output)
            elif self.activation == 1:
                MLP_output = tf.nn.sigmoid(MLP_output)
            elif self.activation == 2:
                MLP_output = tf.nn.tanh(MLP_output)

            A_ = tf.reshape(tf.matmul(MLP_output, self.h), [b, n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            num_idx = tf.reduce_sum(self.num_idx, 1)
            mask_mat = tf.sequence_mask(num_idx, maxlen=n, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keep_dims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return tf.reduce_sum(A * self.embedding_q_, 1)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, self.user_input)  # (b, n, e)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_input)  # (b, 1, e)

            if self.algorithm == 0:
                self.embedding_p = self._attention_MLP(self.embedding_q_ * self.embedding_q)
            else:
                n = tf.shape(self.user_input)[1]
                self.embedding_p = self._attention_MLP(
                    tf.concat([self.embedding_q_, tf.tile(self.embedding_q, tf.stack([1, n, 1]))], 2))

            self.embedding_q = tf.reduce_sum(self.embedding_q, 1)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(self.num_idx, tf.constant(self.alpha, tf.float32, [1]))
            self.output = tf.sigmoid(
                self.coeff * tf.expand_dims(tf.reduce_sum(self.embedding_p * self.embedding_q, 1), 1) + self.bias_i)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                        self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_)) + \
                        self.eta_bilinear * tf.reduce_sum(tf.square(self.W))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            # self.train_step = self.optimizer.minimize(self.loss)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
            self.grads, _ = zip(*grads_and_vars)


    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()

        logging.info("already build the computing graph...")

    # def creat_saver(self):

    def save_model(self, path):
        with self.graph.as_default():
            self.saver.save(self.sess, path)

    def set_params(self, params=None):
        '''set model parameters'''
        if params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, params):
                    variable.load(value, self.sess)

    def get_params(self):
        '''get model parameters'''
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self):
        ''':aet model gradients'''
        with self.graph.as_default():
            return self.sess.run(self.grads)

    def set_gradients(self, grads):
        '''set model gradients'''
        with self.graph.as_default():
            set_grad = self.optimizer.apply_gradients(grads)
            self.sess.run(set_grad)

    def local_training(self, train_data):
        '''to do'''
        # generate data
        user_input, num_idx, item_input, labels = train_data
        # random response
        # labels = np.where(labels == 1, np.random.binomial(1, 1-p, len(labels)), np.random.binomial(1, p, len(labels)))
        feed_dict = {self.user_input: user_input, self.num_idx: num_idx[:, None],
                     self.item_input: item_input[:, None],
                     self.labels: labels[:, None]}
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict=feed_dict)
        return loss

    def training_loss(self, train_data):
        user_input, num_idx, item_input, labels = train_data
        feed_dict = {self.user_input: user_input, self.num_idx: num_idx[:, None],
                     self.item_input: item_input[:, None], self.labels: labels[:, None]}
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict=feed_dict)
        return loss

    def test(self, test_data):
        user_input, num_idx, item_input, labels = test_data
        feed_dict = {self.user_input: user_input, self.num_idx: num_idx[:, None],
                     self.item_input: item_input[:, None], self.labels: labels[:, None]}
        with self.graph.as_default():
            pre, loss = self.sess.run([self.output, self.loss], feed_dict=feed_dict)
        return pre, loss

