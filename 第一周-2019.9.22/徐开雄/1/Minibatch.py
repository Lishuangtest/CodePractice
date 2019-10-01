import torch
import torch.utils.data as Data
torch.manual_seed(1)

BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    #num_workers=2,
)

for epoch in range(3):
    #enumerate,每一步提取添加一个索引
    for step, (batch_x, batch_y) in enumerate(loader):
        #traing……
        print('Epoch', epoch,'| Step:',step,'| batch x:',
              batch_x.numpy(),'|batch y:',batch_y.numpy())
