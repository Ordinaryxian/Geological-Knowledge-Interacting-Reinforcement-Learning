import pandas as pd

fileCoorLabel = "./DataRAW/suizao.csv"
fileNode = "./DataRAW/suizao_Feature.csv"
fileGAE = "./GAE_result/GAE_0.00050.csv"
fileFault = "./DataRAW/suizao_Fault.csv"

df_coor_label = pd.read_csv(fileCoorLabel)
df_node = pd.read_csv(fileNode)
df_gae = pd.read_csv(fileGAE)
df_fault = pd.read_csv(fileFault)

filtered_indices = df_coor_label[df_coor_label['XX'] < 734039].index

filtered_coor_label = df_coor_label.loc[filtered_indices]
filtered_node = df_node.loc[filtered_indices]
filtered_gae = df_gae.loc[filtered_indices]
filtered_fault = df_fault.loc[filtered_indices]

filtered_coor_label.to_csv("./DataGRAPH/suizao_train.csv", index=False)
filtered_node.to_csv("./DataGRAPH/suizao_Feature_train.csv", index=False)
filtered_gae.to_csv("./DataGRAPH/GAE_0.00050_train.csv", index=False)
filtered_fault.to_csv("./DataGRAPH/suizao_Fault_train.csv", index=False)
