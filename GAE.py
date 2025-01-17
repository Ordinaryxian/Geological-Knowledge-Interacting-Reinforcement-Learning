import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.nn
from numpy import var
from osgeo import gdal
from torch import matmul, optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.special import expit



'''构建GAE网络架构'''
class SpatialBranch(nn.Module):
    def __init__(self, input, hidden):
        super().__init__()
        # Encoder
        self.GCN1 = torch_geometric.nn.GCNConv(in_channels=input, out_channels=32)
        self.bn1 = nn.BatchNorm1d(32)
        self.GCN2 = torch_geometric.nn.GCNConv(in_channels=32, out_channels=16)
        self.bn2 = nn.BatchNorm1d(16)
        self.GCN3 = torch_geometric.nn.GCNConv(in_channels=16, out_channels=hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        # Decoder
        self.GCN4 = torch_geometric.nn.GCNConv(hidden, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.GCN5 = torch_geometric.nn.GCNConv(16, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.GCN6 = torch_geometric.nn.GCNConv(32, input)

    def forward(self, x, edge_index, edge_weight):
        x = self.GCN1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.GCN2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.GCN3(x, edge_index, edge_weight)
        x = self.bn3(x)

        x = self.GCN4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.GCN5(x, edge_index, edge_weight)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.GCN6(x, edge_index, edge_weight)

        return x


def accuracy_calculate(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


device = 'cuda'
# device = 'cpu'
torch.cuda.manual_seed(20)
torch.manual_seed(20)
np.random.seed(20)


def EdgeConstruct(StartEntity, EndEntity, StartLoc, EndLoc, LimitDistance=3000, unDirected=True):
    EdgeSet = []
    StartXY = StartEntity[['XX', 'YY']].values
    EndXY = EndEntity[['XX', 'YY']].values
    DistanceSet = []
    for i in StartXY:
        Distance = np.sqrt((i[0] - EndXY[:, 0]) ** 2 + (i[1] - EndXY[:, 1]) ** 2)
        DistanceSet.append(Distance)
    for i in range(len(DistanceSet)):
        for j in range(len(DistanceSet[i])):
            if DistanceSet[i][j] <= LimitDistance:
                EdgeSet.append([StartLoc[i], EndLoc[j], DistanceSet[i][j]])
                if unDirected:
                    EdgeSet.append([EndLoc[j], StartLoc[i], DistanceSet[i][j]])
    return np.array(EdgeSet)


Node = pd.read_csv("./DataRAW/suizao.csv")
Node_Feature = pd.read_csv("./DataRAW/suizao_Feature.csv")
Edge = EdgeConstruct(Node, Node, np.arange(Node.shape[0]), np.arange(Node.shape[0]), 1400, True)
Feature = torch.FloatTensor(Node_Feature.values).to(device)
EdgeIndex = torch.LongTensor(Edge[:, 0:-1].T).to(device)
EdgeWeight = torch.FloatTensor(Edge[:, -1]).to(device)
EdgeWeight = 1 - (EdgeWeight - EdgeWeight.min()) / (EdgeWeight.max() - EdgeWeight.min())


Spatial = SpatialBranch(Node_Feature.shape[1], 8).to(device)

epoch = 1000
learningRate = 0.0005
optimizer = optim.Adam([{'params': Spatial.parameters()}], lr=learningRate, weight_decay=5e-3)
Loss = nn.MSELoss().to(device)

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []


for i in range(epoch):
    Spatial.train()
    optimizer.zero_grad()

    SpatialReconstruct = Spatial(Feature, EdgeIndex, EdgeWeight)

    loss = Loss(SpatialReconstruct, Feature)

    loss.backward()
    optimizer.step()
    print(i, loss.to('cpu').detach().numpy())
    train_loss_list.append(loss.item())

torch.save(Spatial, './GAE_result/GAE_{:.5f}.pth'.format(learningRate))

#绘制损失函数和准确率曲线
x1t = range(0, epoch)
y1t = train_loss_list
plt.plot(x1t, y1t, '-', color='blue', label='Train loss')
plt.xlabel('Iteration', family='Times New Roman', fontsize=12)
plt.ylabel('Loss', family='Times New Roman', fontsize=12)
plt.xticks(fontname="Times New Roman", fontsize=10)
plt.yticks(fontname="Times New Roman", fontsize=10)
plt.legend(['Train loss', 'Test loss'], prop={"family": "Times New Roman", "size": 12})
plt.savefig('./GAE_result/loss_{:.5f}.png'.format(learningRate), dpi=300)
plt.close()

dataframe = pd.DataFrame({'loss': train_loss_list})
dataframe.to_csv("./GAE_result/loss_acc_{:.5f}.csv".format(learningRate), index=False, sep=',')

print('Read prediction dataset')
Node_predict = pd.read_csv("./DataRAW/suizao.csv")
Node_predict_Feature = pd.read_csv("./DataRAW/suizao_Feature.csv")
PointXY = Node_predict[['XX', 'YY']].values.T
Edge_predict = EdgeConstruct(Node_predict, Node_predict, np.arange(Node_predict.shape[0]),
                             np.arange(Node_predict.shape[0]), 1400, True)
Feature_predict = torch.FloatTensor(Node_predict_Feature.values).to(device)
EdgeIndex_predict = torch.LongTensor(Edge_predict[:, 0:-1].T).to(device)

EdgeWeight_predict = torch.FloatTensor(Edge_predict[:, -1]).to(device)
EdgeWeight_norm = 1 - (EdgeWeight_predict - EdgeWeight_predict.min()) / (
        EdgeWeight_predict.max() - EdgeWeight_predict.min())

print('Finish reading')

Spatial = torch.load('./GAE_result/GAE_{:.5f}.pth'.format(learningRate))
Spatial.eval()
SpatialReconstruct = Spatial(Feature_predict, EdgeIndex_predict, EdgeWeight_predict)


Error_temp = (SpatialReconstruct - Feature_predict) ** 2
ReconstructError = np.mean(Error_temp.to('cpu').detach().numpy(), axis=-1)
Prediction = np.array(ReconstructError)


tempXY = []
for i in range(len(PointXY[0])):
    tempXY.append([PointXY[0, i], PointXY[1, i]])
PointXY = np.array(tempXY)
FinalResult = np.append(PointXY, Prediction.reshape(len(Prediction),1), axis=-1)
result_dataframe = pd.DataFrame({'XX': PointXY[:, 0], 'YY': PointXY[:, 1], 'result':FinalResult[:, 2], 'label':Node_predict['label']})
result_dataframe.to_csv("./GAE_result/GAE_{:.5f}.csv".format(learningRate), index=False, sep=',')