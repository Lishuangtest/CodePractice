import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(1)    # reproducible
import matplotlib.pyplot as plt

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plot dataest
plt.scatter(x.numpy(), y.numpy())
plt.show()



