import torch
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
x = Variable(x)

y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()

y_softplus = F.softplus(x).data.numpy() # there's no softplus in torch

y_softmax = torch.softmax(x, dim=0).data.numpy()   #softmax is a special kind of activation function, it is about probability
