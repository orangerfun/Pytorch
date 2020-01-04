## 网络结构
* 最开始为输出通道数为64、步幅为2的7x7卷积层后接步幅为2的3x3的最⼤池化层
```python3
# 起始主网络
net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
					nn.BatchNorm2d(64),
					nn.ReLU(),
					nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```
* 后面再接4个残差块，残差块结构如下：<br>
```python3
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
	if first_block:
		assert in_channels == out_channels
	blk = []
	for i in range(num_residuals):
		if i == 0 and not first_block:
			blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
		else:
			blk.append(Residual(out_channels, out_channels))
	return nn.Sequential(*blk)
  ```
  * 再接全局平均池化层和全连接层<br>
  总体结构如下图所示：<br>
  ![](https://github.com/orangerfun/Pytorch/raw/master/ResNet/ResNet.png)
