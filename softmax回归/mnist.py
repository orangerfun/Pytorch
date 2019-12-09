import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
# sys.path.append("..")
# import d2lzh_pytorch as d2l

# 第一次使用数据集时会自动下载
mnist_train = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST",\
								train = True,\
								download = True,\
								transform = transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root = "./Datasets/FashionMNIST",\
								train = False,\
								download = True,\
								transform = transforms.ToTensor())

print(len(mnist_train),"\n",len(mnist_test))

# 本数据集中共9个标签，将数值标签转换成文本标签
def get_fashion_mnist_labels(labels):
	text_labels=["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker","bag", "ankleboot"]
	return [text_labels[int(i)] for i in labels]


# 在一行中画多张图片和对应标签
def show_fashion_mnist(images, labels):
	_, figs = plt.subplots(1, len(images), figsize=(12,12))
	for f, img, lbl in zip(figs, images, labels):
		f.imshow(img.view((28,28)).numpy())
		f.set_title(lbl)
		f.axes.get_xaxis().set_visible(False)
		f.axes.get_yaxis().set_visible(False)
	plt.show()

# x, y = [], []
# for i in range(10):
# 	x.append(mnist_train[i][0])
# 	y.append(mnist_train[i][1])
# show_fashion_mnist(x, get_fashion_mnist_labels(y))

# 读取小批量数据
batch_size = 256
trainer_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False)

# 查看读取一遍数据需要的时间
start = time.time()
for x,y in trainer_iter:
	continue
print("%.2f sec" % (time.time()-start))
