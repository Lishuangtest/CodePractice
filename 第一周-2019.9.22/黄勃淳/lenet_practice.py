import torch.nn as nn
import torch.nn.functional as F
###########网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
#############损失函数
input = Variable(t.randn(1, 1, 32, 32))
out = net(input)

out.backward(Variable(t.ones(1, 10)))
output = net(input)
target = Variable(t.arange(0, 10))
criterion = nn.MSELoss
loss = criterion(output, target)
net.zero_grad()
loss.backward()
#####优化器
import torch.optim as optim
optimizer = optim.SGD(net.parameter(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
