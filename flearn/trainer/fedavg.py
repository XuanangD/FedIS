# -*- coding: utf-8 -*-
import numpy as np
import logging
from time import time, strftime, localtime
import random

from .fedbase import BaseFederated


class Server(BaseFederated):
    def __init__(self, dataset, model, args):
        print('Using Federated Avg to Train')
        super(Server, self).__init__(dataset, model, args)


    def train(self, args):
        '''Train using Federated Average'''
        # initialize for training batches
        user_index = list(range(len(self.clients)))

        # server train by outer epoch
        print("start training")
        hr_per_round, ndcg_per_round = [], []
        for epoch_count in range(args.epochs):
            cms = []  # buffer for receiving client solutions
            train_loss = []
            train_begin = time()

            # train selected clients
            select_clients = random.sample(user_index, args.frac)
            for index in select_clients:
                self.client_model.set_params(self.global_state)
                # client train by inner epoch
                loss, client_grad = self.clients[index].local_update(self.client_model, args.local_epochs)
                train_loss.append(loss)
                cms.append(client_grad)

            # aggregate
            if args.option == 0:
                self.global_state = self.aggregate_grad(self.global_state, cms)
            else:
                self.global_state = self.aggregate_weight(self.global_state, cms)

            train_loss = np.mean(train_loss)
            train_time = time() - train_begin
            # test global model
            if epoch_count % args.verbose == 0:
                # test loss
                eval_begin = time()
                self.client_model.set_params(self.global_state)
                hr_test, ndcg_test, loss_test = [], [], []
                for idx in user_index:
                    (hr, ndcg, loss) = self.clients[idx].test(self.client_model)
                    hr_test.append(hr)
                    ndcg_test.append(ndcg)
                    loss_test.append(loss)
                hr, ndcg, test_loss = np.array(hr_test).mean(), np.array(ndcg_test).mean(), np.array(loss_test).mean()
                eval_time = time() - eval_begin
                # select_clients = np.array(losses).argsort()[-args.frac:]
                hr_per_round.append(hr)
                ndcg_per_round.append(ndcg)
                logging.info(
                    "Epoch %d [%.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f " % (
                        epoch_count, train_time, hr, ndcg, test_loss, eval_time, train_loss))
                print(
                    "Epoch %d [%.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f " % (
                        epoch_count, train_time, hr, ndcg, test_loss, eval_time, train_loss))
        # save test results
        logging.info("HR:{}".format(",".join(map(str, hr_per_round))))
        logging.info("NDCG:{}".format(",".join(map(str, ndcg_per_round))))
        self.client_model.save_model('./save_model/'+args.dataset +'/'+args.dataset)
