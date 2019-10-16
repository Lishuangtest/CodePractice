#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import misc
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#%%

# 设备设置
if torch.cuda.is_available() == True:
    torch.cuda.set_device(1)  # 这句用来设置pytorch在哪块GPU上运行，pytorch-cpu版本不需要运行这句
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
num_epoch = 5  # 训练5轮
num_class = 10  # 10类数据
BATCH_SIZE = 32  # 每个batch有32个数据
LR = 0.001  # 学习率

#%%
# 分类的标签
# import pickle
# import os

# filepath = './cifar-10-python/cifar-10-batches-py'
# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='latin')
#     return dict

# data = unpickle(filepath+'/batches.meta')
# print(data['label_names'])

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

#%%
transform = transforms.Compose(
    [
        # +4填充至36x36
        transforms.Pad(4),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 随机左右旋转40度
        transforms.RandomRotation(20),
        # 随机裁剪至32x32
        transforms.RandomCrop(32),
        # 转换至Tensor
        transforms.ToTensor(),
        #  归一化
        # transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5)),   # 3 for RGB channels
    ]
)

#%%
# 加载数据
cifar10Path = ".\cifar-10-python"
DOWNLOAD = False

# 训练集合
train_dataset = torchvision.datasets.CIFAR10(
    root=cifar10Path, train=True, transform=transform, download=DOWNLOAD
)  # 这里的trainsform是指的是数据转换方式，因为数据集合数据增强有利于防止过拟合

# 测试集合
test_dataset = torchvision.datasets.CIFAR10(
    root=cifar10Path, train=False, transform=transform
)

# data loader
# 训练数据
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# 测试数据
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# #%%
# DataLoader支持iter的方式访问但是不支持其他方式访问
idx = 12
data_iter = iter(test_loader)
img, label = next(data_iter)
image = img[idx].numpy()
# 因为第一个维度是通道数即为3 第二三个是32 所有要有调换维度的操作
image = np.transpose(image, (1, 2, 0))
plt.imsave('pic1.png',image)
classes[label[idx]]


#%%
# 定义网络
class CNN(nn.Module):
    def __init__(self, num_class=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积层
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            # 批归一化
            nn.BatchNorm2d(16),
            # ReLU 激活
            nn.ReLU(),
            # 池化层:最大池化
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(8 * 8 * 32, num_class)  # ?和下面的reshap有关 搞懂

    # 向前传播
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)  # ?什么意思弄懂
        out = self.fc(out)
        return out


#%%
# 实例化一个模型，并迁移至gpu
model = CNN(num_class).to(device)
# model = CNN(num_class)
#%%
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 使用adam作为优化器

#%%
####################################################################################
total_step = len(train_loader)  #总的训练次数
for epoch in range(num_epoch):
    for i,(x,y) in enumerate(train_loader):
        #数据对迁移到gpu
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        #计算error
        loss = criterion(output,y)
        #反向传播
        optimizer.zero_grad()  #清零梯度
        loss.backward()  #传播
        optimizer.step()  #优化器利用梯度优化
        
        if (i + 1) % 100 == 0:
            #loss.item() 返回的是传播的梯度
            print("Epoch [{}/{}],Step [{}/{}],Loss:{:.4f}".format(epoch+1,num_epoch,i+1,total_step,loss.item()))
#%%
#测试
model.eval()

# 节省计算资源，不去计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
#%%%%%%%%%%%%
#取其中一组
data_iter = iter(test_loader)
images, labels = next(data_iter)

#取出一个batch中的一张图片
image = images[3].numpy()
image = np.transpose(image, (1, 2, 0))
plt.imshow(image)
plt.title(classes[labels[3].numpy()]) #输出真实标签
plt.show()
#%%%%%%%%
#开始测试
imagebatch = image.reshape(-1, 3, 32, 32)  #调整为(B,C,H,W)

#转化为torch tensor
image_tensor = torch.from_numpy(imagebatch)
#模型评估
model.eval()
output = model(image_tensor.to(device))#输出的标签
_, predicted = torch.max(output.data, 1)  #取预测值
pre = predicted.cpu().numpy
print('预测结果')
print(pre)
print(classes[pre[0]])

#%%
