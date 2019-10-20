import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from model1 import NetD,NetG

#argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数
#使用 argparse 的第一步是创建一个 ArgumentParser 对象。
parser=argparse.ArgumentParser()
#给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。
parser.add_argument('--batchSize',type=int,default=64)
parser.add_argument('--imageSize',type=int,default=96)
#help用来描述这个选项的作用,只是描述而已，没其他作用
parser.add_argument('--nz',type=int,default=100)
parser.add_argument('--ngf',type=int,default=64)
parser.add_argument('--ndf',type=int,default=64)
#批量训练参数
parser.add_argument('--epoch',type=int,default=25)
#学习率
parser.add_argument('--lr',type=float,default=0.0002)
#优化器参数
parser.add_argument('--beta1',type=float,default=0.5)
#数据源，也就是data文件夹
parser.add_argument('--data_path',default='data/')
#输出文件夹，也就是imgs文件夹
parser.add_argument('--outf',default='imgs/')
#parse_args()返回两个值：
# options, 这是一个对象（optpars.Values)，保存有命令行参数值。只要知道命令行参数名，如file,就可以访问其对应的值：options.file。
# args，一个由positional arguments组成的列表。
opt=parser.parse_args()
#定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#图像读入与预处理
transfroms=torchvision.transforms.Compose(
    [
        torchvision.transforms.Scale(opt.imageSize),
        torchvision.transforms.ToTensor(),
        #imagenet数据集的话，由于它的图像都是RGB图像，因此他们的均值和标准差各3个，分别对应其R,G,B值。例如([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])就是Imagenet dataset的标准化系数（RGB三个通道对应三组系数）。数据集给出的均值和标准差系数，每个数据集都不同的，都是数据集提供方给出的。这是用来对图像数据做标准化用的。
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ]
)
#将图像数据从数据源中取出，并将其转换成预处理后的图像
dataset=torchvision.datasets.ImageFolder(opt.data_path,transform=transfroms)

dataloader=torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    #是否将数据集中顺序打乱
    shuffle=True,
    #dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
    drop_last=True
)

netG=NetG(opt.ngf,opt.nz).to(device)
netD=NetD(opt.ndf).to(device)

#此为交叉熵损失函数的一种
criterion=nn.BCELoss()
#这边更改了优化器的默认参数，用来优化生成器的网络参数
optimizerG=torch.optim.Adam(netG.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
#这边也更改了优化器的默认参数，用来优化鉴别器的网络参数
optimizerD=torch.optim.Adam(netD.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
#将batchSize转换为FloatTensor类型，并存入label
label=torch.FloatTensor(opt.batchSize)
real_label=1
fake_label=0

#一批训练图片为25张图片，从1-26刚好25步，一步步用图片来训练网络
for epoch in range(1,opt.epoch+1):
    #这一步是用dataloader来优化训练集并且从训练集中取出imgs
    for i,(imgs,_) in enumerate(dataloader):
        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()
        imgs=imgs.to(device)
        output = netD(imgs)
        #让D尽可能的把真图片判别为1
        label.data.fill_(real_label)
        label=label.to(device)
        errD_real=criterion(output,label)
        errD_real.backward()
        label.data.fill_(fake_label)
        noise=torch.randn(opt.batchSize,opt.nz,1,1) #从一批次的图片中生成噪音，并存入opt.nz中
        noise=noise.to(device)
        #用噪音来生成假图
        fake=netG(noise)
        #简单来说detach就是截断反向传播的梯度流。GAN的G的更新，主要是GAN loss。就是G生成的fake图让D来判别，得到的损失，计算梯度进行反传。这个梯度只能影响G，不能影响D！
        output=netD(fake.detach())
        #让D尽可能把假图片判别为0
        errD_fake=criterion(output,label)
        errD_fake.backward()
        errD=errD_real+errD_fake
        #根据损失，来优化鉴别器，就是优化它的参数
        optimizerD.step()

        #固定鉴别器D，训练生成器G。
        optimizerG.zero_grad()
        #让D尽可能把G生成的假图判为1
        label.data.fill_(real_label)
        label=label.to(device)
        output=netD(fake)
        errG=criterion(output,label)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
    torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))










