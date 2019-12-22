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
