##  网络结构
denseNet和resNet结构很相似，如下图所示<br>
![](https://github.com/orangerfun/Pytorch/raw/master/DenseNet/rd.png)
与ResNet的主要区别在于，DenseNet⾥模块B的输出不是像ResNet那样和模块 A的输出相加，⽽是在通道维上连结。这样模块A的输出可以直接传⼊模块B后⾯的层。在这个设计⾥，模块A直接跟模块B后⾯的所有层连接在了⼀起
<br>
程序中的网络结构设计如下图所示：<br>
![](https://github.com/orangerfun/Pytorch/raw/master/DenseNet/DenseNet.png)
在最前面使用一个卷积计算，然后加入4个denseblock, 4个denseblock结构都相同，其中卷积操作的输出通道数有所变化，结构都如denseblock1所示，最后接上全连接层进行预测
