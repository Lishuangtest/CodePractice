import torch
from torch.autograd import Variable


tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad=True)

t_out = torch.mean(tensor*tensor)    # x^2
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

