import pandas as pd
import pickle
from GKIRL_utils import *
from GKIRL_policy import graph_nn
from GKIRL_drl import REINFROCE
from GKIRL_env import graph_env
from torch.distributions import Categorical
from osgeo import gdal


def to_pyg_data(MPMNode, MPMEdge, state_index, MPMEdgeW):
    node_feature=[]
    node_feature.append(MPMNode.iloc[state_index])
    for i in MPMEdge[state_index]:
        node_feature.append(MPMNode.iloc[i])
    edge_index=list(permutations([i for i in range(len(node_feature))],2))[:len(node_feature)-1]
    node_feature=torch.tensor(node_feature,dtype=torch.float)
    edge_index=torch.tensor(edge_index,dtype=torch.long)
    edge_index=edge_index.t().contiguous()
    weight=MPMEdgeW[state_index]
    weight=torch.tensor(weight,dtype=torch.float)
    state=Data(x=node_feature,edge_index=edge_index,weight=weight)
    return state

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
seed = randomseed
seed_torch(seed)
fileNode = "./DataRAW/suizao_Feature.csv"
fileEdge = "./DataGRAPH/suizao_edge.pkl"
fileEdgeW = "./DataGRAPH/suizao_edge_weights.pkl"


policy_predicted = torch.load('GKIRL.pth',map_location=torch.device('cpu'))
policy_predicted = policy_predicted.to(device)
MPM_coordinate = pd.read_csv('./DataRAW/suizao.csv')
MPMNode = pd.read_csv(fileNode, header=0)
with open(fileEdge, 'rb') as fE:
    MPMEdge = pickle.load(fE)
with open(fileEdgeW, 'rb') as fEW:
    MPMEdgeW = pickle.load(fEW)
result = pd.DataFrame()
prob = []
for i in range(MPMNode.shape[0]):
    state_graph = to_pyg_data(MPMNode, MPMEdge, i, MPMEdgeW)
    action_prob = policy_predicted(state_graph.x.to(device), state_graph.edge_index.to(device), state_graph.weight.to(device))
    max_index = torch.argmax(action_prob[0])
    action1 = 1
    prob.append(action_prob[0][action1].cpu().item())
result['XX'] = MPM_coordinate['XX']
result['YY'] = MPM_coordinate['YY']
result['label'] = MPM_coordinate['label']
result['result'] = prob
result.to_csv('GKIRL_result.csv',index=False, header=True)