import torch
import numpy as np

# abs
data = [-1, -2, 1, 2]
tensor =torch.FloatTensor(data)    # 32bit

print(
    '\nsin',
    '\nnumpy ', np.mean(data),
    '\ntorch ', torch.mean(tensor)
)
