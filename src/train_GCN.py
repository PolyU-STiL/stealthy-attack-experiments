import torch
import numpy as np
import argparse
import os.path as osp
import torch.nn.functional as F
import random
import pickle as pkl
from deeprobust.graph.data import Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from pGRACE.dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer'])
parser.add_argument('--seed', type=int, default=39788)
parser.add_argument('--perturb', action="store_true")
parser.add_argument('--attack_method', type=str, default=None)
parser.add_argument('--attack_rate', type=float, default=0.10)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

torch_seed = args.seed
torch.manual_seed(torch_seed)
random.seed(12345)

device = torch.device(args.device)

path = osp.expanduser('dataset')
path = osp.join(path, args.dataset)
dataset = get_dataset(path, args.dataset)
data = dataset[0]
data = Pyg2Dpr(dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

if args.perturb:
    perturbed_adj = pkl.load(open('poisoned_adj/%s_%s_%f_adj.pkl' % (args.dataset, args.attack_method, args.attack_rate), 'rb')).to(device)

# Setup GCN Model
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)

if args.perturb:
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
else:
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=200, verbose=True)

model.eval()
# You can use the inner function of model to test
model.test(idx_test)
