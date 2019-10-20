import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import  torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
Epoch=2
transform=transforms.Compose(
   [transforms.ToTensor(),
    # transforms.Normalize(mean=(.5,.5,.5),std=(.5,.5,.5))
    ]
)
train_set=torchvision.datasets.CIFAR10(download=False,transform=transform,train=True,root="./root")
trainloader=Data.DataLoader(train_set,batch_size=4,num_workers=2,shuffle=True)
test_set=torchvision.datasets.CIFAR10(download=False,transform=transform,train=False,root="./root")
testloader=Data.DataLoader(test_set,num_workers=2,batch_size=4,shuffle=True)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(img):
    img=img*0.5+0.5
    nping=img.numpy()
    plt.imshow(np.transpose(nping,(1,2,0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5 )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net=Net()
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9)
for epoch in range(Epoch):
    if __name__ == '__main__':
        for i,data in enumerate(trainloader):
            inputs,labels=data
            optimizer.zero_grad()
            output=net(inputs)
            # print(labels.size())
            loss=loss_func(output,labels)
            loss.backward()
            optimizer.step()
            print(loss.data)
if __name__ == '__main__':
    dataiter=iter(trainloader)
    dataiter1=iter(testloader)
    images,labels=dataiter.next()
    images1,labels1=dataiter1.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images1))
    print(' '.join(classes[labels1[j]]for j in range(4)))
    outputs=net(images)
    predict=torch.max(outputs.data,1)[1]
    print(' '.join(classes[predict[j]]for j in range(4)))








