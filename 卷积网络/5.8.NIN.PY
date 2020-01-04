import torch
from torch import nn, optim
import time
import torch.nn.functional as F
from VGG import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
	blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
						nn.ReLU(),
						nn.Conv2d(out_channels, out_channels, kernel_size=1),
						nn.ReLU(),
						nn.Conv2d(out_channels, out_channels, kernel_size=1),
						nn.ReLU())
	return blk


class GlobalAvgPool2d(nn.Module):
	def __init__(self):
		super(GlobalAvgPool2d,self).__init__()
	def forward(self, x):
		return F.avg_pool2d(x, kernel_size=x.size()[2:])    #x.size=[batch_size, channels, wide ,hight], 取高和宽即可

net = nn.Sequential(nin_block(1,96,11,4,0),
					nn.MaxPool2d(kernel_size=3, stride=2),
					nin_block(96, 256, 5, 1, 2),
					nn.MaxPool2d(kernel_size=3, stride=2),
					nin_block(256, 384, 3, 1, 1),
					nn.MaxPool2d(kernel_size=3, stride=2),
					nn.Dropout(0.5),
					nin_block(384, 10, 3, 1, 1),      # 标签类别数为10
					GlobalAvgPool2d(),                # 全局最大池化，shape = [batchsize, channels(10), 1, 1]
					FlattenLayer())                   # 将四维的输出转成二维输出，shape:[batch_size,10]

# 测试，打印每个block中输出的shape
# x = torch.rand(1,1,224,224)
# for name, blk in net.named_children():
# 	x = blk(x)
# 	print(name, "output_shape:", x.shape)

# 获取数据和训练模型
batch_size=128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr = 0.1
num_epochs = 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


