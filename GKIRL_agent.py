import matplotlib.pyplot as plt
import pandas as pd

from GKIRL_utils import *
from GKIRL_policy import graph_nn
from GKIRL_drl import REINFROCE
from GKIRL_env import graph_env
from torch.distributions import Categorical

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
seed = 10
seed_torch(seed)
'''参数设置，读取的文件'''
fileCoorLabel = "./DataGRAPH/suizao_train.csv"
fileNode = "./DataGRAPH/suizao_Feature_train.csv"
fileEdge = "./DataGRAPH/suizao_edge_train.pkl"
fileEdgeW = "./DataGRAPH/suizao_edge_weights_train.pkl"
fileGAE = "./DataGRAPH/GAE_0.00050_train.csv"
fileSpatial = "./DataGRAPH/suizao_neighbor_spatial_train.pkl"
fileDistacne = "./DataGRAPH/suizao_property_distance_train.pkl"
fileFault = "./DataGRAPH/suizao_Fault_train.csv"
g_p = 0.5
learning_rate=0.0005
episodes = 500
len_episode = 100
warm_episode = 5
update_interval = episodes/50
gamma = 0.9
batch_size = 32
num_batches = 64
max_memory = 10000
policy_action_space=2
lamda = 1

env = graph_env(fileCoorLabel=fileCoorLabel,fileNode=fileNode,
                fileEdge=fileEdge, fileEdgeW=fileEdgeW,
                fileGAE=fileGAE,
                fileSpatial=fileSpatial, fileDistance=fileDistacne,
                fileGeo1=fileFault, g_p=g_p, lamda=lamda)

policy = graph_nn(action_space=policy_action_space, input_dim=pd.read_csv(fileNode).shape[1]).to(device)
target_policy = graph_nn(action_space=policy_action_space, input_dim=pd.read_csv(fileNode).shape[1]).to(device)
target_policy.load_state_dict(policy.state_dict())

learner = REINFROCE(policy=policy, target_policy=target_policy,
                    learning_rate=learning_rate, gamma=gamma,
                    batch_size=batch_size, num_batches=num_batches,
                    max_memory=max_memory)

sum_reward = 0
sum_reward_list = []
sum_action = 0
sum_action_list = []
print_interval = 10
episodes_list = []
eps = 1

for episode in range(episodes):
    if episode%print_interval==0 and episode!=0:
        print("# of episode :{}, avg score : {}".format(episode,sum(sum_reward_list[-print_interval:])/print_interval))
    state_graph = env.reset()
    done = False
    step = 0
    while not done:
        action_prob = policy(state_graph.x.to(device), state_graph.edge_index.to(device), state_graph.weight.to(device))
        eps_prob = [eps, 1-eps]
        elements = [True, False]
        result = random.choices(elements, eps_prob)[0]
        if result:
            action = torch.randint(0,action_prob.size(-1),(1,))
        else:
            action = torch.argmax(action_prob, dim=-1)
        next_state, reward, done, _ = env.step(action.item())
        if step>=len_episode:
            done=True
        learner.memory_data((state_graph, action, reward, next_state, done))
        state_graph = next_state
        sum_reward += reward
        sum_action += action.item()
        step += 1
    if episode > warm_episode:
        learner.learn()
    if episode%update_interval == 0 and episode!=0:
        target_policy.load_state_dict(policy.state_dict())
    if episode < 0.9*episodes:
        eps = 1 - (episode/episodes) * 0.95
    else:
        eps = 0.05
    sum_reward_list.append(sum_reward)
    sum_reward=0
    sum_action_list.append(sum_action)
    sum_action=0
    episodes_list.append(episode)

torch.save(policy, 'GKIRL.pth')