# -*- coding: utf-8 -*-
import numpy as np
import math
from flearn.utils.batch_gen import get_train_batch,  load_test


class Client(object):
    def __init__(self, id, train_data, test_data, args):
        self.id = id
        self.train_data = train_data
        self.train_num = len(self.train_data[1])
        self.test_data = load_test(self.id, train_data, test_data)
        # self.model = model
        # self.local_state = self.model.get_params()
        self.args = args

    def set_params(self, model, params):
        '''set model parameters'''
        return model.set_params(params)

    def get_params(self, model):
        '''get model parameters'''
        return model.get_params()

    def get_gradients(self, model):
        '''get model gradient'''
        return model.get_gradients()

    def local_update(self, model, num_epoch):
        '''Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''
        init_state = model.get_params()
        train_loss = []
        for i in range(num_epoch):
            train_data = get_train_batch(self.id, self.train_data, self.args.num_neg)
            train_data = list(train_data)
            train_data[-1] = self.randomize_response(train_data[-1], self.args.epsilon)
            loss = model.local_training(train_data)
            train_loss.append(loss)

        local_state = model.get_params()
        pseudo_grads = [self.dropout(w-w0, self.args.sparse_rate) for w, w0 in zip(local_state, init_state)]
        return np.mean(train_loss), (self.train_num, pseudo_grads)

    def dropout(self, x, keep_prob=0.5):
        mask = np.random.binomial(1, p=1.-keep_prob, size=x.shape)
        x = x * mask
        # x = x / keep_prob
        return x

    def eps2p(self, epsilon, n=2):
        return np.e ** epsilon / (np.e ** epsilon + n - 1)

    def randomize_response(self, bitarray, epsilon):
        p = self.eps2p(epsilon/2) / (self.eps2p(epsilon/2) + 1)
        # q = 1 / (self.eps2p(epsilon/2) + 1)
        return np.where(bitarray == 1, np.random.binomial(1, 1-epsilon, len(bitarray)), np.random.binomial(1, epsilon, len(bitarray)))

    def train_error(self):
        ''' training loss'''
        train_data = get_train_batch(self.id, self.train_data, self.args.num_neg)
        return self.model.training_loss(train_data)

    def test(self, model):
        '''tests current model on local eval_data'''
        predictions, loss = model.test(self.test_data)
        neg_predict, pos_predict = predictions[:-1], predictions[-1]
        position = (neg_predict >= pos_predict).sum()

        # calculate HR@10, NDCG@10, AUC
        hr = position < 10
        ndcg = math.log(2) / math.log(position + 2) if hr else 0
        return (hr, ndcg, loss)
