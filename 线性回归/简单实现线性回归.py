import numpy as np
import torch
import random
import torch.utils.data as Data
from torch import nn

# 生成数据
num_examples = 1000
input_len = 2
w_true = torch.tensor([1.4, 2.9])
b_true = torch.ones(1, dtype=torch.float)
# gamma = torch.tensor(np.random.normal(0, 0.1, (1,num_examples)))
x = torch.tensor(np.random.normal(0,1,(num_examples, input_len)), dtype= torch.float)
y = w_true[0]*x[:,0] + w_true[1]*x[:,1] + b_true
y = y+torch.tensor(np.random.normal(0, 0.1, size=y.size()), dtype = torch.float)

# 读取数据
# 使用torch提供的包来读取数据
batch_size = 10
# 将特征和label组合
dataset = Data.TensorDataset(x, y)
# 读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle = True)

# 尝试打印
# for x, y in data_iter:
# 	print(x, y)
# 	break

# 定义模型
class LinearNet(nn.Module):
	def __init__(self, n_feature):
		super(LinearNet, self).__init__()
		self.linear = nn.Linear(n_feature, 1)

	# 定义前向传播
	def forward(self, x):
		y = self.linear(x)
		return y
		
net = LinearNet(input_len)
# 打印网络结构
print(net)

