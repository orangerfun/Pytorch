import torch
from torch import nn
# 使用Module类
class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.hidden = nn.Linear(784, 256)
		self.act = nn.ReLU()
		self.output = nn.Linear(256, 10)

	def forward(self, x):
		a = self.act(self.hidden(x))
		return self.output(a)
# net = MLP()
# print(net)


# 2.使用Sequential子类
net = nn.Sequential(nn.Linear(784, 256),\
				nn.ReLU(),\
				nn.Linear(256, 10))
# print(net)
# print(net[-1])


# 3.ModuleList子类
net = nn.ModuleList([nn.Linear(786,254), nn.ReLU()])
net.append(nn.Linear(254, 10))  # 可以像列表一样操作
# print(net)


# ModuleDict类
net = nn.ModuleDict({"linear":nn.Linear(784, 256), "act":nn.ReLU()})
# 增加
net["output"]=nn.Linear(256,10)
# print(net.output)
# print(net)

# 继承Module类
class FancyMLP(nn.Module):
	def __init__(self):
		super(FancyMLP,self).__init__()
		self.rand_wieght=torch.rand((20,20),requires_grad=False) #该参数不可训练
		self.linear = nn.Linear(20, 20)

	def forward(self, x):
		x = self.linear(x)
		x = nn.functional.relu(torch.mm(x, self.rand_wieght.data)+1)
		# 复用全连接层，相当于两个全连接层共享参数
		x = self.linear(x)
		# 控制流
		while x.norm().item()>1:    #norm L2范数
			x /= 2
		if x.norm().item()<0.8:
			x *= 10
		return x.sum()

# 初始化模型参数
from torch.nn import init
net = FancyMLP()
# way1
for param in net.parameters():
	init.normal_(param, mean=0, std=0.01)
# way2
for name, param in net.named_parameters():
	if "weight" in name:
		init.normal_(param, mean=0, std=0.01)
		print(name, param.data)

# 自定义初始化模型
def init_weight_(tensor):
	"""
	args:
		tensor:输入x
	"""
	with torch.no_grad():
		# 令权重有⼀半概率初始化为0，有另⼀半概率初始化为[-10,-5]和[5,10]两个区间⾥均匀分布的随机数
		tensor.uniform_(-10,10)
		tensor *= (tensor.abs()>=5).float()
for param in net.parameters():
	init_weight_(param)

