import torch
import time
import torch.nn.functional as F
from torch import nn, optim
from ResNet import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_block(in_channels, out_channels):
	blk = nn.Sequential(nn.BatchNorm2d(in_channels),
						nn.ReLU(),
						nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
	return blk


class DenseBlock(nn.Module):
	def __init__(self, num_convs, in_channels, out_channels):
		super(DenseBlock, self).__init__()
		net = []
		for i in range(num_convs):
			in_c = in_channels + i*out_channels
			net.append(conv_block(in_c, out_channels))
		self.net = nn.ModuleList(net)
		self.out_channels = in_channels + num_convs*out_channels

	def forward(self, x):
		for blk in self.net:
			y = blk(x)
			x = torch.cat((x,y), dim=1)
		return x

# 过度层；用1x1卷积减少通道，平均池化层减半高和宽
def transition_block(in_channels, out_channels):
	blk = nn.Sequential(nn.BatchNorm2d(in_channels),
						nn.ReLU(),
						nn.Conv2d(in_channels, out_channels, kernel_size=1),
						nn.AvgPool2d(kernel_size=2, stride=2))
	return blk

net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
					nn.BatchNorm2d(64),
					nn.ReLU(),
					nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]


# 网络中添加DenseBlock
for i, num_convs in enumerate(num_convs_in_dense_blocks):
	DB = DenseBlock(num_convs, num_channels, growth_rate)
	net.add_module("DenseBlock_%d"%i, DB)
	# 上一个稠密块的输出通道数
	num_channels = DB.out_channels
	if i!=len(num_convs_in_dense_blocks)-1:
		net.add_module("transition_block_%d"%i, transition_block(num_channels, num_channels//2))
		num_channels = num_channels//2


# 网络中增加全局池化层和全连接层
net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", GlobalAvgPool2d())  #outshape=[batch, num_channels, 1, 1]
net.add_module("fc", nn.Sequential(FlattenLayer(),nn.Linear(num_channels, 10)))

if __name__ == '__main__':
	batch_size = 256
	train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
	lr, num_epochs = 0.001, 5
	optimizer = optim.Adam(net.parameters(), lr=lr)
	train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
	



