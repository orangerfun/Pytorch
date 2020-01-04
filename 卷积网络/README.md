# 1.基本API
```
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
  args : in_channels: 输入通道数
         out_channels: 输出通道数 也相当于num_kernels
         kernel_size: 卷积核形状,长宽不同时用tuple, 相同时用int
         padding:输入的每一条边补充0的层数
         stride:步长，长宽不一样时用tuple
         
  input.shape=[batch_size, channels, height1, width1]
  output.shape=[batch_size, out_channels, height2, width2]
 ```
 ```
 torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
  args: kernel_size: max pooling的窗口大小，可以为tuple，在nlp中tuple用更多
        stride: 步长;默认值是kernel_size,可以是tuple
        padding: 输入的每一条边补充0的层数
```
# 2.卷积网络比较
### 2.1 LeNet
使用两个卷积层，每个卷积层后接最大池化层，每个卷积层都使用5x5的窗口，在输出上使用sigmoid激活函数，第⼀个卷积层输出通道数为6，第⼆个卷积层输出通道数则增加到16；最大池化层窗口大小为2x2，步幅也为2；卷积后接3个全连接层，最后一个全连接层用于分类.LeNet网络结构如下：<br>
```python3
self.conv = nn.Sequential(nn.Conv2d(1, 6, 5),\
								nn.Sigmoid(),\
								nn.MaxPool2d(2, 2),\
								nn.Conv2d(6, 16, 5),\
								nn.Sigmoid(),\
								nn.MaxPool2d(2, 2))
 ```
 ### 2.2 AlexNet
 AlexNet使⽤了8层卷积神经⽹络,中有5层卷积和2层全连接隐藏层，以及1个全连接输出层;AlexNet第⼀层中的卷积窗⼝形状是11x11;第⼆层中的卷积窗⼝形状减⼩到5x5,之后全采⽤3x3; 第⼀、第⼆和第五个卷积层之后都使⽤了窗⼝形状为3x3 、步幅为2的最⼤池化层;AlexNet将sigmoid激活函数改成了更加简单的ReLU激活函数;AlexNet通过丢弃法来控制全连接层的模型复杂度; AlexNet引⼊了⼤量的图像增⼴，如翻转、裁剪和颜⾊变化，从⽽进⼀步扩⼤数据集来缓解过拟合, AlexNet网络结构如下所示：
 ```python3
 self.conv = nn.Sequential(nn.Conv2d(1, 96, 11,4),  #in_channels, out_channels, kernel_size, stride, padding
								nn.ReLU(),
								nn.MaxPool2d(3,2), #kernel_size, stride
								nn.Conv2d(96, 256, 5, 1, 2),
								nn.ReLU(),
								nn.MaxPool2d(3,2),
								nn.Conv2d(256, 384, 3, 1, 1),
								nn.ReLU(),
								nn.Conv2d(384, 384, 3, 1, 1),
								nn.ReLU(),
								nn.Conv2d(384, 256, 3, 1, 1),
								nn.ReLU(),
								nn.MaxPool2d(3,2)
								)
		self.fc = nn.Sequential(nn.Linear(256*5*5, 4096),
								nn.ReLU(),
								nn.Dropout(0.5),
								nn.Linear(4096, 4096),
								nn.ReLU(),
								nn.Dropout(0.5),
								nn.Linear(4096, 10))
 ```
 ### 2.3 VGG
 VGG提出了可以通过重复使⽤简单的基础块来构建深度模型的思路；VGG块的组成规律是：连续使⽤数个相同的填充为1、窗⼝形状为3x3的卷积层后接上⼀个步幅为2、窗⼝形状为2x2的最⼤池化层。卷积层保持输⼊的⾼和宽不变，⽽池化层则对其减半;VGG⽹络由卷积层模块后接全连接层模块构成;构造如下结构VGG网络：<br>
它有5个卷积块，前2块使⽤单卷积层，⽽后3块使⽤双卷积层。第⼀块的输⼊输出通道分别是1（因为下⾯要使⽤的Fashion-MNIST数据的通道数为1）和64，之后每次对输出通道数翻倍，直到变为512。因为这个⽹络使⽤了8个卷积层和3个全连接层，所以经常被称为VGG-11<br>
具体网络结构见程序`5.7VGG.py`

### 2.4 NiN
NiN提出了另外⼀个思路，即串联多个由卷积层和“全连接”层构成的⼩⽹络来构建⼀个深层⽹络; 1x1的卷积层可以看作是全连接层，其中通道代表特征，宽高中的每个元素代表样本，具体见`动手学深度学习5.3节`；NiN用1x1的卷积代替全连接层,NiN网络结构如下：<br>

	卷积层-->1x1卷积层-->卷积层-->1x1卷积层
普通卷积网络结构如下：<br>

	卷积层-->卷积层-->全链接层-->全连接层
如下定义了NiN块， 由一个卷积层加两个充当全连接层的1x1卷积层串联而成：<br>
```python3
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
	blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
						nn.ReLU(),
						nn.Conv2d(out_channels, out_channels, kernel_size=1),
						nn.ReLU(),
						nn.Conv2d(out_channels, out_channels, kernel_size=1),
						nn.ReLU())
	return blk
```
NiN的卷积层设定和AlexNet有类似之处，NiN使⽤卷积窗⼝形状分别为11x11, 5x5,和3x3的卷积层，相应的输出通道数也与AlexNet中的⼀致。每个NiN块后接⼀个步幅为2、窗⼝形状为3x3的最⼤池化层。除使⽤NiN块以外，NiN还有⼀个设计与AlexNet显著不同：NiN去掉了AlexNet最后的3个全连接层，取⽽代之地，NiN使⽤了输出通道数等于标签类别数的NiN块然后使⽤全局平均池化层对每个通道中所有元素求平均并直接⽤于分类;全局平均池化层即窗⼝形状等于输⼊空间维形状的平均池化层;NiN的这个设计的好处是可以显著减⼩模型参数尺⼨，从⽽缓解过拟合。然⽽，该设计有时会造成获得有效模型的训练时间的增加<br>
NiN网络结构如下：<br>
```python3
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
```
**小结**<br>
* NiN重复使⽤由卷积层和代替全连接层的1x1卷积层构成的NiN块来构建深层⽹络
* NiN去除了容易造成过拟合的全连接输出层，⽽是将其替换成输出通道数等于标签类别数的NiN
块和全局平均池化层

### 2.5 GoogleNet
GoogLeNet中的基础卷积块叫作Inception块,结构如下图所示：<br>
![](https://github.com/orangerfun/Pytorch/raw/master/5.卷积神经网络/inception.png)

Inception块⾥有4条并⾏的线路。前3条线路使⽤窗⼝⼤⼩分别是1x1, 3x3 和5x5的卷积层来抽取不同空间尺⼨下的信息，其中中间2个线路会对输⼊先做1x1卷积来减少输⼊通道数，以降低模型复杂度。第四条线路则使⽤3x3最⼤池化层，后接1x1卷积层来改变通道数。4条线路都使⽤了合适的填充来使输⼊与输出的⾼和宽⼀致。最后我们将每条线路的输出在通道维上连结，并输⼊接下来的层中去<br>
`Inception`的网络结构如下：<br>
```python3
class Inception(nn.Module):
	def __init__(self, in_c, c1, c2, c3, c4):
		super(Inception, self).__init__()
		# 线路1, 单1x1卷积层
		self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
		# 线路2, 1x1卷积后接3x3卷积层
		self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
		self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
		# 线路3, 1x1卷积后接5x5卷积层
		self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
		self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
		# 线路4,3x3最大化池化层后接1x1卷积层
		self.p4_1 = nn.MaxPool2d(kernel_size = 3, stride=1, padding=1)
		self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
```

GoogLeNet在主体卷积部分中使⽤5个模块每个模块之间使⽤步幅为2的3x3最⼤池化层来减⼩输出⾼宽,第⼀模块使⽤⼀个64通道的 7x7 卷积层,第二，第三...block具体结构见程序`5.9GoogleNet.py`<br>
**小结**
* Inception块是⼀个有4条线路的⼦⽹络。它通过不同窗⼝形状的卷积层和最⼤池化层来并⾏抽取信息，并使⽤1x1卷积层减少通道数从⽽降低模型复杂度
* GoogLeNet将多个设计精细的Inception块和其他层串联起来。其中Inception块的通道数分配之⽐是在ImageNet数据集上通过⼤量的实验得来的



