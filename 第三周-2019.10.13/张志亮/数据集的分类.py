import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from scipy import misc
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
if torch.cuda.is_available==True:
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
device=torch.device("cuda")
num_epoch=5
num_class=10
BATCH_SIZE=32
LR=0.001
classes=("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck",)
transform=transforms.Compose(
    [
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ]
)
cifar10Path="cifar-10-batches-bin"

train_dataset=torchvision.datasets.CIFAR10(
    root=cifar10Path,train=True,transform=transform,download=False
)
test_dataset=torchvision.datasets.CIFAR10(
    root=cifar10Path,train=False,transform=transform
)
train_loader=torch.utils.data.DataLoader(
    dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)
idx=10
data_iter=iter(test_loader)
img,label=next(data_iter)
image=img[idx].numpy()
image=np.transpose(image,(1,2,0))
plt.imsave('pic1.png',image[idx])
classes[label[idx]]
class CNN(nn.Module):
    def __init__(self,num_class=10):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.fc=nn.Linear(8*8*32,num_class)
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        return out
model=CNN(num_class).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
total_step=len(train_loader)
for epoch in range(num_epoch):
    for i,(x,y) in enumerate(train_loader):
        x=x.to(device)
        y=y.to(device)
        output=model(x)
        loss=criterion(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100==0:
            print("Epoch [{}/{}],Step [{}/{}],Loss:{:.4f}".format(epoch + 1, num_epoch, i + 1, total_step, loss.item()))
model.eval()
with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))








