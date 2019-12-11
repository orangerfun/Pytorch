在pytorch中进行L2正则化，只需设置optimizer中的`weight_decay`参数即可；weight_decay相当于L2正则中的epsilong<br>

    optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=0.1)
