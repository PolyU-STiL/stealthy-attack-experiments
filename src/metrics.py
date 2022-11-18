import io

import numpy as np
import pickle
import argparse
import os
import sys

import torch


def arg_parse():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('--exp', type=str, default='1', help='experiment folder')
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora or CiteSeer')
    parser.add_argument('--attack', type=str, default='CLGA', help='attack algorithm')
    parser.add_argument('--budget', type=str, default='01', help='attack budget(%)')
    parser.add_argument('--metric', type=str, default='balanced_homophily', help='metric algorithm')
    args = parser.parse_args()
    return args

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def get_matrix(args):
    matrix_path = '../poisoned_graph/experiment{}/{}_{}_0.{}0000_adj.pkl'.format(args.exp, args.dataset,
                                                                                 args.attack,
                                                                                 args.budget)  # ../poisoned_graph/experiment1/Cora_CLGA_0.010000_adj.pkl
    exist = os.path.exists(matrix_path)
    if exist:
        f = open(matrix_path, 'rb')
        # file = torch.load(matrix_path, map_location=torch.device('cpu'))
        matrix = CPU_Unpickler(f).load()

        return np.array(matrix)
        # f = open(matrix_path, 'rb')
        # file = pickle.load(f)
        # matrix = np.array(file, dtype=np.int64)
        # return matrix


        # if (args.attack=='metattack') | (args.attack=='minmax') | (args.attack=='pgd'):
        #     f = open(matrix_path, 'rb')
        #     # file = torch.load(matrix_path, map_location=torch.device('cpu'))
        #     matrix = CPU_Unpickler(f).load()
        #
        #     return np.array(matrix)
        # else:
        #     f = open(matrix_path, 'rb')
        #     file = pickle.load(f)
        #     matrix = np.array(file, dtype=np.int64)
        #     return matrix
    else:
        return -1


def get_label(args):
    lowerCase = 'cora' if args.dataset == 'Cora' else 'citeseer'
    ally_path = './dataset/{}/Citation/{}/raw/ind.{}.ally'.format(args.dataset, args.dataset, lowerCase)
    ty_path = './dataset/{}/Citation/{}/raw/ind.{}.ty'.format(args.dataset, args.dataset, lowerCase)
    idx_path = './dataset/{}/Citation/{}/raw/ind.{}.test.index'.format(args.dataset, args.dataset, lowerCase)
    if args.dataset == 'Cora':
        node_n = 2708
    elif args.dataset == 'CiteSeer':
        node_n = 3327
    fy = open(ally_path, 'rb')
    ally = np.array(pickle.load(fy))

    fty = open(ty_path, 'rb')
    ty = np.array(pickle.load(fty))

    idx = np.loadtxt(idx_path)
    idx = np.array(idx, dtype=np.int64)
    # print(idx.max())

    label = np.zeros([node_n]) - 1
    # initialize label
    for i in range(node_n):
        if i < 1708:
            label[i] = ally[i].argmax()
        else:
            for j in range(len(idx)):
                if idx[j] == i:
                    label[i] = ty[j].argmax()
    return np.array(label, dtype=int)


def edge_homophily(matrix, label):
    count = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if j <= i:
                continue
            else:
                if (matrix[i][j] == 1) & (label[i] == label[j]):
                    count += 1
    medges = matrix.sum() / 2
    eh = count / medges
    return eh


def node_homophily(matrix, label):
    n = matrix.shape[0]
    count = 0.0
    for i in range(matrix.shape[0]):
        nh_per_node = 0.0
        dv = matrix[i].sum()
        if dv == 0:  # if a isolated node, define it's nh_per_node as 0
            continue
        label_v = label[i]
        for j in range(matrix.shape[1]):
            if (matrix[i][j] == 1) & (label[i] == label[j]):
                nh_per_node += 1
        nh_per_node /= dv
        count += nh_per_node

    nh = count / n
    return nh


def class_homophily(matrix, label):
    C = label.max()
    c_h = 0
    for k in range(C + 1):
        sum_of_node_k = 0
        n_k = 0
        degree_sum = 0
        for i in range(label.shape[0]):
            if label[i] == k:  # for each yv == k
                n_k += 1
                degree_sum += matrix[i].sum()
                for j in range(matrix[i].shape[0]):
                    if matrix[i][j]:
                        if label[j] == k:
                            sum_of_node_k += 1
        temp = sum_of_node_k / degree_sum - n_k / matrix.shape[0]
        c_h += max(0, temp)
    c_h /= C
    return c_h


def D_k(matrix, label, k):
    dk = 0
    for i in range(label.shape[0]):
        if label[i] == k:  # for each yv == k
            dk += matrix[i].sum()
    return dk


def adjusted_homophily(matrix, label):
    eh = edge_homophily(matrix, label)
    e = label.shape[0]
    c = label.max()
    temp = 0
    for k in range(c + 1):
        temp += pow(D_k(matrix, label, k) / (2 * e), 2)
    ah = (eh - temp) / (1 - temp)
    return ah


def balanced_homophily(matrix, label):
    c = label.max()
    h_ball = 0
    for k in range(c + 1):
        dk = D_k(matrix, label, k)
        temp = 0
        for i in range(label.shape[0]):
            if label[i] == k:
                for j in range(matrix[i].shape[0]):
                    if j <= i:
                        continue
                    if (matrix[i][j] == 1) & (label[j] == k):
                        temp += 1
        h_ball += (temp / dk)
    h_ball /= c
    bh = (h_ball - 1 / c) * c / (c - 1)
    return bh


if __name__ == '__main__':
    args = arg_parse()
    matrix = get_matrix(args)
    label = get_label(args)

    if type(matrix) != type(-1):
        if args.metric == 'edge_homophily':
            metric_result = edge_homophily(matrix, label)
        elif args.metric == 'node_homophily':
            metric_result = node_homophily(matrix, label)
        elif args.metric == 'class_homophily':
            metric_result = class_homophily(matrix, label)
        elif args.metric == 'adjusted_homophily':
            metric_result = adjusted_homophily(matrix, label)
        elif args.metric == 'balanced_homophily':
            metric_result = balanced_homophily(matrix, label)
        print(metric_result)
        # print("dataset={}, attack={}, budget={}%,metric={}, result= {}".format(args.dataset, args.attack, args.budget,
        #                                                                        args.metric, metric_result))
    else:
        print(" ")
