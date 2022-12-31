# -*- coding: utf-8 -*-
import numpy as np
import logging

from flearn.model.client import Client


class BaseFederated(object):
    def __init__(self, dataset, model, args):
        self.client_model = model
        self.args = args
        self.clients = self.setup_clients(dataset, self.client_model)
        self.global_state = self.client_model.get_params()
        logging.info('{} Clients in Total'.format(len(self.clients)))
        print('{} Clients in Total'.format(len(self.clients)))


    def setup_clients(self, dataset, model):
        # users, groups, train_data, test_data = dataset
        # if len(groups) == 0:
        #     groups = [None for _ in users]
        all_clients = [Client(i, dataset.train_data[i], dataset.test_data[i], self.args) for i in range(dataset.num_users)]

        return all_clients

    def test(self):
        """tests self.latest_model on given clients
        """
        return

    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0] * len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    # simpleAVg
    def aggregate_weight(self, model, wsolns):
        total_weight = 0.0
        base = [0] * len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return list(map(lambda x: x[0]+x[1], zip(averaged_soln, model)))

    # reptile
    def aggregate_grad(self, model, wsolns):
        total_weight = 0.0
        base = [0] * len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight = 1
            for i, v in enumerate(soln):
                base[i] += 1 * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return list(map(lambda x: self.args.glo_lr*x[0]+x[1], zip(averaged_soln, model)))