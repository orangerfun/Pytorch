import torch
from torch import nn

def corr2d(X,K):
	'''
	args:
		x:输入数组
		k:核数组
	func:二维相互关运算（卷积运算）
	'''
	h, w = K.shape
	Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			Y[i,j]=(X[i:i+h, j:j+w]*K).sum()
	return Y

# 测试上面函数：
# X = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
# K = torch.tensor([[0,1],[2,3]])
# print(corr2d(X,K))

# 自定义一个二维卷积层
class Conv2D(nn.Module):
	def __init__(self, kernel_size):
		super(Conv2D, self).__init__()
		self.weight = nn.Parameter(torch.randn(kernel_size))
		self.bias = nn.Parameter(torch.randn(1))
	def forward(self, x):
		return corr2d(x, self.weight) + self.bias
