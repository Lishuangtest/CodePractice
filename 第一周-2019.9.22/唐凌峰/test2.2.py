import torch
from torch.autograd import Variable
from torch import Tensor

tensor = torch.FloatTensor([[1,2],[3,4]])         #通过Tensor来对Varible初始化
variable = Variable(tensor, requires_grad=True)   #注意require_grad的传递规律，只要所依赖的张量中一个为True则所有的都为True

v_out = torch.mean(variable*variable)  # x^2 = 7.5
v_out.backward()                       # backpropagation from v_out

print(variable.grad)
variable.data.numpy()   # numpy format，与tensor不同，variable需要使用属性data，返回一个tensor类再numpify，而新版本使用使用tensor即可，算是优化
