import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将数据转成tensor，标准化均值和方差到[-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 下载数据集与处理数据
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
print('ddd',trainset.class_to_idx)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size = 4,
                                          shuffle = True,
                                        ) 
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train = False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=True,
                                         )
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


'''
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)       # 每次迭代取的是一个batch
images, labels = dataiter.next()   # 如果batch_size为4，则取出来的images是4*32*32*3的tensor，labels是1×4的向量

# show images
imshow(torchvision.utils.make_grid(images))     # 将images中的4张图片拼成一张图片
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''



# 网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = Net().to(device)


# 损失函数和优化器
criterion = nn.CrossEntropyLoss()    #CrossEntropyLoss函数输入一个概率向量，输出向量对应的标签
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练
if __name__ == "__main__":
    net.train()
    for epoch in range(20):
        sum_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            if i%2000 == 1999:
                print('[%d,%5d] loss : %.3f]' % (epoch+1, i+1, sum_loss/2000))
                sum_loss = 0.0
    #torch.save(net, 'D:\python程序\spyder程序\cifar10_model.pth')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
