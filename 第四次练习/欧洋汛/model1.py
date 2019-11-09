#反卷积过程参考这篇博客：https://blog.csdn.net/qq_27261889/article/details/86304061
import torch.nn as nn
#定义生成器网络
class NetG(nn.Module):
    #这边的ngf为生成器生成的feature map数，nz为噪声的维度，即我们可以认为输入的噪声为一个1*1*nz的特征图
    def __init__(self,ngf,nz):
        super(NetG,self).__init__()
        #layer1输入的是一个100*1*1的一个随机噪声，输出尺寸为(ngf*8)x4x4的特征图,ngf表示数字
        self.layer1=nn.Sequential(
            #逆卷积ConvTranspose2d
            nn.ConvTranspose2d(nz,ngf*8,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(ngf*8),
            #inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.ReLU(inplace=True)
        )
        #layer输出尺寸（ngf*4）*8*8
        #Height′=Height+(Stride−1)∗(Height−1),不用加padding
        #新的卷积核：Stride′=1这个数不变，无论你输入是什么。kernel的size也不变,padding为Size−padding−1.size为输入的卷积核的size
        #Height=(Height+2∗padding−kernel)/strides′+1
        self.layer2=nn.Sequential(
            #这边的卷积核个数比起上面那一层的减少一半，以此类推
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
        )
        #layer输出尺寸为（ngf*2）*16*16
        self.layer3=nn.Sequential(
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        #layer4输出尺寸（ngf）*32*32
        self.layer4=nn.Sequential(
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
        )
        # layer5输出尺寸 3x96x96,这边就直接说明白了吧，ndf就是一个数字，这边就只拿3层卷积核来进行生成图像了
        self.layer5=nn.Sequential(
            nn.ConvTranspose2d(ngf,3,5,3,1,bias=False),
            nn.Tanh()
        )

    #定义NetG的前向传播
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        return out

#定义鉴别器网络
class NetD(nn.Module):
    def __init__(self,ndf):
        super(NetD,self).__init__()
        #layer1输入3*96*96，输出（ndf）*32*32
        self.layer1=nn.Sequential(
            nn.Conv2d(3,ndf,kernel_size=5,stride=3,padding=1,bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,inplace=True)
        )
        #layer2输出（ndf*2）*16*16
        self.layer2=nn.Sequential(
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        #layer3输出（ndf*4）*8*8
        self.layer3=nn.Sequential(
            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #layer4输出(ndf*8)x4x4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True)
        )
        #把(ndf*8)*4*4的特征图变成一个一维向量，然后根据该一维向量算出该图片是生成还是原图片
        self.layer5=nn.Sequential(
            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid(),
        )

    # 定义NetD的前向传播
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        return out







