# 1. 继承MODULE类来构造模型
```python3
class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.hidden = nn.Linear(784, 256)
		self.act = nn.ReLU()
		self.output = nn.Linear(256, 10)

	def forward(self, x):
		a = self.act(self.hidden(x))
		return self.output(a)
# net = MLP()
```
# 2. Sequential 类
```python3
net = nn.Sequential(nn.Linear(784, 256),\
				nn.ReLU(),\
				nn.Linear(256, 10))
```
# 3. ModuleList子类
ModuleList 接收⼀个⼦模块的**列表**作为输⼊，然后也可以类似List那样进⾏append和extend操作
```python3
net = nn.ModuleList([nn.Linear(786,254), nn.ReLU()])
net.append(nn.Linear(254, 10))  # 可以像列表一样操作
```
# 4. ModuleDict 类
ModuleDict 接收⼀个⼦模块的字典作为输⼊, 然后也可以类似字典那样进⾏添加访问操作:
```python3
net = nn.ModuleDict({"linear":nn.Linear(784, 256), "act":nn.ReLU()})
# 增加
net["output"]=nn.Linear(256,10)
```
# 5.总结
虽然 `2、3、4中Sequential` 等类可以使模型构造更加简单，但直接继承 Module 类可以极⼤地拓展模型构
造的灵活性
```python3
class FancyMLP(nn.Module):
	def __init__(self):
		super(FancyMLP,self).__init__()
		self.rand_wieght=torch.rand((20,20),requires_grad=False) #该参数不可训练
		self.linear = nn.Linear(20, 20)

	def forward(self, x):
		x = self.linear(x)
		x = nn.functional.relu(torch.mm(x, self.rand_wieght.data)+1)
		# 复用全连接层，相当于两个全连接层共享参数
		x = self.linear(x)
		# 控制流
		while x.norm().item()>1:    #norm L2范数
			x /= 2
		if x.norm().item()<0.8:
			x *= 10
		return x.sum()
net = FancyMLP()
```
# 6.初始化模型参数
## 6.1 默认初始化
定义好模型net后，torch会自动初始化参数，因此也可直接使用默认参数
## 6.2 nn.init.noraml_()
```python3
# 初始化模型参数
from torch.nn import init
net = FancyMLP()
# way1
for param in net.parameters():
	init.normal_(param, mean=0, std=0.01)
# way2
for name, param in net.named_parameters():
	if "weight" in name:
		init.normal_(param, mean=0, std=0.01)
		print(name, param.data)
```
## 6.2自定义初始化
```python3
def init_weight_(tensor):
	"""
	args:
		tensor:输入x
	"""
	with torch.no_grad():
		# 令权重有⼀半概率初始化为0，有另⼀半概率初始化为[-10,-5]和[5,10]两个区间⾥均匀分布的随机数
		tensor.uniform_(-10,10)
		tensor *= (tensor.abs()>=5).float()
for param in net.parameters():
	init_weight_(param)
```

# 7.保存模型
**保存**
```python3
torch.save(net.state_dict(),PATH)
```
**加载**
```python3
model = ModelClass(args)
model.load_state_dict(torch.load(path)
```
