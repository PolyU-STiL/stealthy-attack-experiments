import numpy as np
import pickle
import argparse
import os
import sys

parser = argparse.ArgumentParser(description=globals()['__doc__'])
# diffusion models
parser.add_argument('--dataset', type=str, default='Cora', help='Cora or CiteSeer')
parser.add_argument('--attack', type=str, default='CLGA', help='attack algorithm')
parser.add_argument('--budget', type=str, default='1', help='attack budget(%)')
parser.add_argument('--metric', type=str, default='node_homophily', help='metric algorithm')
args = parser.parse_args()

matrix_dir_path = '../poisoned_graph/{}/{}/{}%'.format(args.attack, args.dataset, args.budget)  # ./CLGA/Cora/1%

for root, dirs, files in os.walk(matrix_dir_path):
    file_name = files[0]

ally_path = './dataset/Cora/Citation/Cora/raw/ind.cora.ally'
ty_path = './dataset/Cora/Citation/Cora/raw/ind.cora.ty'
idx_path = './dataset/Cora/Citation/Cora/raw/ind.cora.test.index'

f = open(os.path.join(matrix_dir_path, file_name), 'rb')
matrix = np.array(pickle.load(f), dtype=int)

fy = open(ally_path, 'rb')
ally = np.array(pickle.load(fy))

fty = open(ty_path, 'rb')
ty = np.array(pickle.load(fty))

idx = np.loadtxt(idx_path)
idx = np.array(idx, dtype=np.int)
# print(idx.max())

label = np.zeros([2708]) - 1
# initialize label
for i in range(matrix.shape[0]):
    if i < 1708:
        label[i] = ally[i].argmax()
    else:
        for j in range(len(idx)):
            if idx[j] == i:
                label[i] = ty[j].argmax()

# metric evaluate
if args.metric == 'edge_homophily':
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
    print("dataset={}, attack={}, budget={}%,metric={}, result= {}".format(args.dataset, args.attack, args.budget,
                                                                           args.metric, eh))
if args.metric == 'node_homophily':
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
    print("dataset={}, attack={}, budget={}%,metric={}, result= {}".format(args.dataset, args.attack, args.budget,
                                                                           args.metric, nh))
