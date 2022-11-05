import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer'])
parser.add_argument('--seed', type=int, default=15)
parser.add_argument('--perturb', action="store_true")
parser.add_argument('--attack_method', type=str, default=None)
parser.add_argument('--attack_rate', type=float, default=0.10)

device = torch.device(args.device)

path = osp.expanduser('dataset')
path = osp.join(path, args.dataset)
dataset = get_dataset(path, args.dataset)
data = dataset[0]

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)
# Or we can just use setting='prognn' to get the splits
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

if args.perturb:
    perturbed_adj = pkl.load(open('poisoned_adj/%s_%s_%f_adj.pkl' % (args.dataset, args.attack_method, args.attack_rate), 'rb')).to(device)

# Setup GCN Model
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)

model.fit(features, perturbed_adj, labels, idx_train, train_iters=200, verbose=True)
# # using validation to pick model
# model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
model.eval()
# You can use the inner function of model to test
model.test(idx_test)
