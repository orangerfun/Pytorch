import numpy as np
import torch
import random
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import torch.optim as optim

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

# 定义模型/方法1
# class LinearNet(nn.Module):
# 	def __init__(self, n_feature):
# 		super(LinearNet, self).__init__()
# 		self.linear = nn.Linear(n_feature, 1)

# 	# 定义前向传播
# 	def forward(self, x):
# 		y = self.linear(x)
# 		return y
		
# net = LinearNet(input_len)
# # 打印网络结构
# print(net)

# 定义模型/方法2
# net = nn.Sequential(nn.Linear(input_len,1))
# print(net)
# print(net[0])

# 定义模型/方法3
net = nn.Sequential()
net.add_module("linear", nn.Linear(input_len, 1))

# 初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03) #net.parameters() 来查看模型所有的可学习参数，此函数将返回⼀个⽣成器
# print(optimizer)

# 调整学习率
for param_group in optimizer.param_groups:
	param_group["lr"] *= 0.1

# 训练模型
num_epochs = 10
for epoch in range(1, num_epochs+1):
	for x,y in data_iter:
		output = net(x)
		l = loss(output, y.view(-1,1))
		optimizer.zero_grad() # 梯度清零
		l.backward()
		optimizer.step()
	print("epoch:%d\tloss:%f"%(epoch, l.item()))

print("========学习的参数==========")
print(net[0].weight)
print(net[0].bias)
print("=========真实参数===========")
print(w_true)
print(b_true)

