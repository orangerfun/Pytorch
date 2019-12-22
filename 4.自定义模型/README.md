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
# 5.
