import gym
from gym import spaces
import torch
from torch_geometric.data import Data
from itertools import permutations
import pandas as pd
import pickle
import random
import numpy as np

# 定义基于图的环境模型
class graph_env(gym.Env):
    def __init__(self, fileCoorLabel,fileNode, fileEdge, fileEdgeW, fileGAE, fileSpatial, fileDistance,fileGeo1, g_p, lamda):
        super(graph_env, self).__init__()
        self.MPMNode = pd.read_csv(fileNode, header=0)
        self.fileCoorLabel = pd.read_csv(fileCoorLabel)

        self.label_1 = self.fileCoorLabel[self.fileCoorLabel['label']==1]
        self.label_0 = self.fileCoorLabel[self.fileCoorLabel['label']==0]
        with open(fileEdge, 'rb') as fE:
            self.MPMEdge = pickle.load(fE)
        with open(fileEdgeW, 'rb') as fEW:
            self.MPMEdgeW = pickle.load(fEW)
        with open(fileSpatial, 'rb') as fS:
            self.data_index_pkl = pickle.load(fS)
        with open(fileDistance, 'rb') as fD:
            self.property_distace = pickle.load(fD)
        self.reward_GAE = pd.read_csv(fileGAE, header=0)
        if fileGeo1 == 0:
            self.reward_Geo1 = 0
        else:
            self.reward_Geo1 = pd.read_csv(fileGeo1, header=0)
        self.g_p = g_p
        self.data_index = 0
        self.lamda = lamda
    def to_pyg_data(self, state_index):
        node_feature = []
        node_feature.append(self.MPMNode.iloc[state_index])
        for i in self.MPMEdge[state_index]:
            node_feature.append(self.MPMNode.iloc[i])
        edge_index=list(permutations([i for i in range(len(node_feature))],2))[:len(node_feature)-1]
        node_feature = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor(edge_index,dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        weight = self.MPMEdgeW[state_index]
        weight = torch.tensor(weight, dtype=torch.float)
        state = Data(x=node_feature, edge_index=edge_index, weight=weight)
        return state
    def reset(self):
        random_index = self.label_1.sample(n=1).index[0]
        self.state_index = random_index
        state = self.to_pyg_data(self.state_index)
        return state
    def step(self, action):
        reward = self.get_reward(action)
        state = self.next_state(action)
        done = False
        others = {}
        return state, reward, done, others

    def get_reward(self, action):
        if self.reward_GAE['label'].iloc[self.state_index] == 1:
            r_GAE=self.reward_GAE['result'].iloc[self.state_index]
            if action == 1:
                r_deposit = 1
            else:
                r_deposit = -1
        else:
            r_GAE = self.reward_GAE['result'].iloc[self.state_index]
            if action == 0:
                r_deposit = 0
            else:
                r_deposit = -1
        r_constraint = self.reward_Geo1['Fault'].iloc[self.state_index]
        return r_deposit+r_GAE+self.lamda*r_constraint
    def next_state(self, action):
        index=np.random.choice(['d_l','d_u'], p=[self.g_p,1-self.g_p])
        if index=='d_l':
            random_index=self.label_1.sample(n=1).index[0]
            self.state_index=random_index
            state=self.to_pyg_data(self.state_index)
        else:
            if action == 1:
                max_value = max(self.property_distace[self.state_index])
                max_index = self.property_distace[self.state_index].index(max_value)
                self.state_index = self.data_index_pkl[self.state_index][max_index]
                state = self.to_pyg_data(self.state_index)
            if action == 0:
                min_value=min(self.property_distace[self.state_index])
                min_index=self.property_distace[self.state_index].index(min_value)
                self.state_index=self.data_index_pkl[self.state_index][min_index]
                state=self.to_pyg_data(self.state_index)
        return state