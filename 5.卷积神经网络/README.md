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


