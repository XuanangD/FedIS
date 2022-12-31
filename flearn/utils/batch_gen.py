# -*- coding: utf-8 -*-
import numpy as np


def get_train_batch(id, train_data, num_neg):
    ''' generate train data batch'''
    train_matrix, train_list = train_data
    user_input, item_input, labels = [], [], []
    num_items = train_matrix.shape[1]
    for i in train_list:
        # positive instance
        user_input.append(id)
        item_input.append(i)
        labels.append(1)
        # negative instance
        for t in range(num_neg):
            j = np.random.randint(num_items)
            while j in train_list:
                j = np.random.randint(num_items)
            user_input.append(id)
            item_input.append(j)
            labels.append(0)
    user_list, num_list, item_list, labels_list = [], [], [], []
    batch_index = list(range((1 + num_neg) * len(train_list)))
    np.random.shuffle(batch_index)
    for idx in batch_index:
        user_idx = user_input[idx]
        item_idx = item_input[idx]
        nonzero_row = []
        nonzero_row += train_list
        num_list.append(_remove_item(num_items, nonzero_row, item_idx))
        user_list.append(nonzero_row)
        item_list.append(item_idx)
        labels_list.append(labels[idx])
    user_input = np.array(_add_mask(num_items, user_list, max(num_list)))
    num_idx = np.array(num_list)
    item_input = np.array(item_list)
    labels = np.array(labels_list)
    return (user_input, num_idx, item_input, labels)

def load_test(id, train_data, test_data):
    ''' load eval data_dict '''
    train_matrix, train_list = train_data
    test_rating, test_negatives = test_data
    user = train_list
    items = test_negatives
    items.append(test_rating[1])
    num_item = len(train_list)
    num_idx = np.full(len(items),num_item, dtype=np.int32 )
    user_input = []
    for i in range(len(items)):
        user_input.append(user)
    user_input = np.array(user_input)
    item_input = np.array(items)
    labels = np.zeros(len(test_negatives))
    labels[-1] = 1
    return (user_input, num_idx, item_input, labels)

def _remove_item(feature_mask, users, item):
    flag = 0
    for i in range(len(users)):
        if users[i] == item:
            users[i] = users[-1]
            users[-1] = feature_mask
            flag = 1
            break
    return len(users) - flag

def _add_mask(feature_mask, features, num_max):
    # uniformalize the length of each batch
    for i in range(len(features)):
        features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
    return features