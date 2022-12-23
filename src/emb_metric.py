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
import sys
from scipy.special import rel_entr
import io
import os
import math

def arg_parse():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--exp', type=str, default='1', help='experiment folder')
    parser.add_argument('--attack', type=str, default='CLGA', help='attack algorithm')
    parser.add_argument('--budget', type=str, default='01', help='attack budget(%)')
    parser.add_argument('--metric', type=str, default='prox1', choices=['prox1', 'prox2','jsd','ln'])

    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer'])
    parser.add_argument('--seed', type=int, default=39788)
    # parser.add_argument('--perturb', action="store_true")

    parser.add_argument('--device', type=str, default='cuda:7')

    args = parser.parse_args()
    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)
    return args

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_embb(path):
    exist = os.path.exists(path)
    if exist:
        f = open(path, 'rb')
        # file = torch.load(matrix_path, map_location=torch.device('cpu'))
        matrix = CPU_Unpickler(f).load()
        perturbed_adj = np.array(matrix)

    # if args.perturb:
    # perturbed_adj = pkl.load(open(path,'rb') ).to(device)

    # Setup GCN Model
    model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
    # train 140, val 500, test 1000

    model.eval()



    model.test(idx_test)
    #embedding

    embb= model.output_test()
    # print(embb)
    sm = torch.nn.Softmax(dim=1)
    sm_embb = sm(embb)
    return perturbed_adj, sm_embb 

def get_clean():
    g = pkl.load(open('./dataset/Cora/Citation/Cora/raw/ind.cora.graph','rb') )
    clean_adj=np.zeros([2708,2708])
    for i, (k, v) in enumerate(g.items()):
        clean_adj[k][v]=1
        # print('源节点:', k, '邻居节点:', v)
    return clean_adj

def prox1(p_adj, sm_embb, labels):
    ''' Return size : (node_num,)'''
    prox1=np.zeros_like(labels, dtype=np.float32)

    for i in range(prox1.shape[0]):
        embbi = sm_embb[i]
        count=0
        prox_temp=0
        for j in range (prox1.shape[0]):
            if(p_adj[i][j]!=0): #neighbor j
                count+=1
                embbj = sm_embb[j]
                prox_temp += sum(rel_entr(embbi, embbj))
        if count == 0:
            count =1
        prox1[i] = prox_temp/count

    return prox1

def prox2(p_adj, sm_embb, labels):
    ''' Return size : (node_num,)'''
    prox2 = np.zeros_like(labels).astype(np.float32)
    for i in range (prox2.shape[0]):
        count=0
        prox_temp=0
        for j in range(prox2.shape[0]):
            if (p_adj[i][j]!=0):
                count+=1
                embbj = sm_embb[j]
                for k in range(prox2.shape[0]): # need to be symetric because kl(j,k) != kl(k,j)
                    if(p_adj[i][k]!=0):
                        embbk = sm_embb[k]
                        prox_temp += sum(rel_entr(embbj, embbk))
        if(count==1):
            count_1=1
        elif(count ==0):
            count =1
            count_1 =1
        else:
            count_1=count-1
        # print(count)
        # print(prox_temp)
        # print(prox_temp/(count*count_1))
        prox2[i] = prox_temp/(count*count_1)
        # print(prox2[i])
        # sys.exit()
    return prox2

class JSD():
    def jsd_H(self,pi):
        # print(pi.shape)
        res=0
        for i in pi:
            if i !=0:
                res+= (i * math.log(i))
        return -1 * res
    def JSD_res(self,p_adj, sm_embb, labels):
        jsd=np.zeros_like(labels, dtype=np.float32)
        for i in range (labels.shape[0]):
            left=np.zeros([sm_embb.shape[1]],dtype=np.float32)
            right=0
            count=0
            for j in range(labels.shape[0]):
                if (p_adj[i][j]!=0):
                    count +=1
                    left+=sm_embb[j]
                    right+=self.jsd_H(sm_embb[j])
            if count ==0:
                count =1
            right =right/count
            left = left/count
            left = self.jsd_H(left)
            jsd[i]=left - right
        return jsd

class LN():
    def LN_dis(self,pi,pj):
        # print(pi.shape)
        res=0
        for i in range(pi.shape[0]):
            if(pj[i]!=0):
                res+= (pi[i] * math.log(pj[i]))
        return -1 * res
    def LN_res(self,p_adj, sm_embb, labels):
        ln=np.zeros_like(p_adj, dtype=np.float32)-1
        for i in range (labels.shape[0]):
            n_i= p_adj[i].sum()
            for j in range (labels.shape[0]):
                if(p_adj[i][j]==0):
                    n_j=p_adj[j].sum()
                    ln[i][j]=self.LN_dis(sm_embb[i],sm_embb[j])/((n_i+1)*(n_j+1))
                    print(i,j)
        return ln

def generate_result(labels, metric):
    if metric!='ln':
        res=np.zeros([7,5,labels.shape[0]],dtype=np.float32)
    else:
        res=np.zeros([7,5,labels.shape[0],labels.shape[0]],dtype=np.float32)
    # print(res.shape)
    # sys.exit()
    if labels.shape[0] ==2708:
        name = 'Cora'
    else:
        name= 'CiteSeer'
    # for i,att in enumerate(["CLGA"]):
    for i,att in enumerate(["CLGA","dice", "metattack", "minmax", "nodeembeddingattack", "pgd", "random"]):
        for j, budget in enumerate(["01", "05", "10", "15", "20"]):
            path = '../poisoned_graph/experiment1/{}_{}_0.{}0000_adj.pkl'.format( name, att, budget)
            p_adj, sm_embb = get_embb(path)
            # p_adj=p_adj.cpu().detach().numpy() # convert to np
            sm_embb=sm_embb.cpu().detach().numpy()
            print(att,budget,labels.shape[0])
            if metric =='prox1':
                res[i][j] = prox1(p_adj, sm_embb, labels)
            elif metric == 'prox2':
                res[i][j] = prox2(p_adj, sm_embb, labels)
            elif metric == 'jsd':
                jsd=JSD()
                jsdr=jsd.JSD_res(p_adj=p_adj, sm_embb=sm_embb, labels=labels)
                res[i][j] = jsdr
            elif metric == 'ln':
                ln=LN()
                lnr=ln.LN_res(p_adj, sm_embb, labels)
                res[i][j]=lnr
    with open("../data/{}_{}.pkl".format(name,metric), 'wb') as fout:
      pkl.dump(res, fout)




if __name__ == '__main__':
    args=arg_parse()
    device = torch.device(args.device)

    

    clean_adj=get_clean()
    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)

    path="./dataset/Cora"
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = Pyg2Dpr(dataset)
    adj, features, labels = data.adj, data.features, data.labels

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    generate_result(labels, args.metric)
    # sys.exit()

    
    # matrix_dir_path = '../poisoned_graph/experiment{}/{}_{}_0.{}0000_adj.pkl'.format(args.exp,  args.dataset, args.attack, args.budget)  # ../poisoned_graph/experiment1/Cora_CLGA_0.010000_adj.pkl
    # p_adj, sm_embb = get_embb(matrix_dir_path)
    
    # # p_adj=p_adj.cpu().detach().numpy() # convert to np
    # # p_adj=clean_adj
    
    # sm_embb=sm_embb.cpu().detach().numpy()
    # # print(sm_embb.shape)
    # # jsd=JSD()
    # # jsdr=jsd.JSD_res(p_adj=p_adj, sm_embb=sm_embb, labels=labels)
    # ln=LN()
    # lnr=ln.LN_res(p_adj, sm_embb, labels)
    # print(ln.shape)
    # print(lnr[0].max())

    # # p1=prox2()
    # # print(p1.mean(),p1.std())


