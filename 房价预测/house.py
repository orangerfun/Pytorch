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
def log_rmse(net, features, labels):
	with torch.no_grad():
		clipped_preds = torch.max(net(features), torch.tensor(1.0))
		rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()).mean())
	return rmse.item()


# 训练函数
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
	train_ls, test_ls = [], []
	dataset = torch.utils.data.TensorDataset(train_features, train_labels)
	train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
	optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
	net = net.float()
	for epoch in range(num_epochs):
		for x, y in train_iter:
			y_hat = net(x)
			l = loss(y_hat, y)
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
		train_ls.append(log_rmse(net, train_features, train_labels))
		if test_labels is not None:
			test_ls.append(log_rmse(net, test_features, test_labels))
	return train_ls, test_ls


# 定义第i折交叉验证时所需的训练数据和验证数据
def get_k_fold_data(k, i, x, y):
	assert k > 1
	fold_size = x.shape[0]//k
	x_train, y_train = None, None
	for j in range(k):
		idx = slice(j*fold_size, (j+1)*fold_size)
		if j == i:
			x_valid, y_valid = x[idx,:], y[idx]
		elif x_train is None:
			x_train, y_train = x[idx,:], y[idx]
		else:
			x_train = torch.cat((x_train, x[idx,:]), dim=0)
			y_train = torch.cat((y_train, y[idx]), dim=0)
	return x_train, y_train, x_valid, y_valid


# K折交叉验证，训练K次， 返回训练和验证平均误差
def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
	train_l_sum, valid_l_sum = 0, 0
	for i in range(k):
		data = get_k_fold_data(k, i, x_train, y_train)
		net = get_net(x_train.shape[1])
		train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
		train_l_sum += train_ls[-1]
		valid_l_sum += valid_ls[-1]
		print("fold:%d, train_rmse:%f, valid_rmse: %f"%(i, train_ls[-1], valid_ls[-1]))
	return train_l_sum/k, valid_l_sum/k


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
	net = get_net(train_features.shape[1])
	train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
	print("train rmse %f"%train_ls[-1])
	preds = net(test_features).detach().numpy()
	test_data["SalePrice"]=pd.Series(preds.reshape(1,-1)[0])
	submission = pd.concat([test_data["Id"],test_data["SalePrice"]], axis=1)
	submission.to_csv("./submission.csv", index=False)



if __name__ == '__main__':
	# 使用交叉验证来选择模型超参数
	k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
	# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
	# print("%d-flod validation: avg train rmse %f,avg valid rmse %f"%(k, train_l, valid_l))

	train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
