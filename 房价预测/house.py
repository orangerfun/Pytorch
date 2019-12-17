import numpy as np
import torch
from torch import nn
import pandas as pd
torch.set_default_tensor_type(torch.FloatTensor)

data_path = "./data/"
train_data = pd.read_csv(data_path+"train.csv")
test_data = pd.read_csv(data_path+"test.csv")

# 数据预处理
all_features = pd.concat((train_data.iloc[:,1:-1], test_data.iloc[:,1:]))
numeric_features = all_features.dtypes[all_features.dtypes !="object"].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x-x.mean())/(x.std()))
all_features = all_features.fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values,dtype = torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1,1)

# 损失函数
loss = torch.nn.MSELoss()

# 构建模型
def get_net(feature_num):
	net = nn.Linear(feature_num, 1)
	for param in net.parameters():
		nn.init.normal_(param, mean=0, std=0.01)
	return net

# 定义均方根误差，评价模型

