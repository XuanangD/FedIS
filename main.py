# -*- coding: utf-8 -*-
import logging
import os
import argparse
from time import strftime, localtime
import numpy as np
import random

from utils.Dataset import Dataset
from flearn.model.nais import NAIS
from flearn.trainer.fedavg import Server


def parse_args():
    parser = argparse.ArgumentParser(description="Run NAIS.")
    # federated arguments
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--frac', type=float, default=100,
                        help='Fraction of clients: C')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Number of local epochs: E.')
    parser.add_argument('--epsilon', type=float, default=0.,
                        help='Random response on labels.')
    parser.add_argument('--sparse_rate', type=float, default=0.,
                       help='Model update sparse rate.')
    parser.add_argument('--option', type=float, default=0,
                        help='0 for Reptile aggregation, 1 for simpleAvg.')
    parser.add_argument('--glo_lr', type=float, default=0.9,
                        help='Learning rate.')
    # model arguments
    parser.add_argument('--model', type=str, default='nais',
                        help='Name of model.')
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
    parser.add_argument('--weight_size', type=int, default=16,
                        help='weight size.')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--data_alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7,1e-5]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=1,
                        help='Caculate training loss or nor')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Index of coefficient of sum of exp(A)')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--activation', type=int, default=0,
                        help='Activation for ReLU, sigmoid, tanh.')
    parser.add_argument('--algorithm', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')


    return parser.parse_args(args=[])


if __name__ == '__main__':

    args = parse_args()
    print(args)

    log_dir = "Log/%s/" % args.dataset
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, "%s_%d_T%d_E%d_K%d_%s" %
                                              (args.dataset, args.option, args.epochs, args.local_epochs, args.frac,
                                               strftime('%Y%m%d%H%M', localtime()))),
                        level=logging.INFO)
    logging.info("begin training %s model ......" % args.model)
    logging.info(args)

    dataset = Dataset(args.path + args.dataset)
    model = NAIS(dataset.num_items, args)

    server = Server(dataset, model, args)
    server.train(args)
