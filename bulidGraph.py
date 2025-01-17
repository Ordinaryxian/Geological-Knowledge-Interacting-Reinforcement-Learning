import pickle as pkl
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram

# train = ''
train = '_train'

if train == '':
    data = pd.read_csv("./DataRAW/suizao%s.csv"%train)
    xy = data[['XX', 'YY']].values
    feature = pd.read_csv("./DataRAW/suizao_Feature%s.csv"%train)
if train == '_train':
    data = pd.read_csv("./DataGRAPH/suizao%s.csv"%train)
    xy = data[['XX', 'YY']].values
    feature = pd.read_csv("./DataGRAPH/suizao_Feature%s.csv"%train)

tree = cKDTree(xy)


num_neighbors = 9*9
num_neighbors = num_neighbors - 1
distances, indices = tree.query(xy, k=num_neighbors + 1)
edge = {i: list(ind[1:]) for i, ind in enumerate(indices)}
with open('./DataGRAPH/suizao_edge%s.pkl'%train, 'wb') as f:
    pkl.dump(edge, f)

min_dist = np.min(distances[:, 1:], axis=1)
max_dist = np.max(distances[:, 1:], axis=1)

min_dist = min_dist[:, np.newaxis]
max_dist = max_dist[:, np.newaxis]

normalized_distances = 1 - (distances[:, 1:] - min_dist) / (max_dist - min_dist)

edge_weights = {}
for i, dists in enumerate(normalized_distances):
    edge_weights[i] = list(dists)
with open('./DataGRAPH/suizao_edge_weights%s.pkl'%train, 'wb') as f:
    pkl.dump(edge_weights, f)


if train == '_train':
    with open('./DataGRAPH/suizao_edge%s.pkl'%train, 'rb') as f:
        suizao_edge=pkl.load(f)

    df_label_1=data[data['label']==1]
    df_label_0=data[(data['label']==0) | (data['label']==-9999)]
    neighbor_spatial ={}
    property_distance = {}
    for idx in data.index:
        x1,y1=data.iloc[idx]['XX'],data.iloc[idx]['YY']

        distances=np.sqrt((df_label_0['XX']-x1)**2+(df_label_0['YY']-y1)**2)
        close_points=df_label_0[distances<=3000]
        neighbor_spatial[idx] = close_points.index[1:]

        graphs =[]
        for neighbor_idx in close_points.index:
            a = suizao_edge[neighbor_idx].copy()
            a.insert(0, neighbor_idx)
            graph = Graph({0:list(range(1,len(suizao_edge[neighbor_idx])+1))},
                            node_labels={i:tuple(feature.iloc[a[i]]) for i in range(len(a))})
            graphs.append(graph)

        wl_kernel = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram)
        kernel_matrix = wl_kernel.fit_transform(graphs)
        similarity_list = []
        for i in range(len(close_points.index[1:])):
            similarity_list.append(kernel_matrix[0][i+1])
        property_distance[idx] = similarity_list

    with open('./DataGRAPH/suizao_neighbor_spatial%s.pkl'%train,'wb') as f:
        pkl.dump(neighbor_spatial,f)

    with open('./DataGRAPH/suizao_property_distance%s.pkl'%train,'wb') as f:
        pkl.dump(property_distance,f)