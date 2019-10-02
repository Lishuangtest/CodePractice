#%%
import torch
import torch.utils.data as Data  # batch trainning moudle

# 表示一个批次有5个数据进行训练
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
print(x)
print(y)
#%%
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=3,  # 两个进程进行提取
)

for eopch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        #training
        print(
            "eopch: ",
            eopch,
            "| step:",
            step,
            "| batch x: ",
            batch_x.numpy(),
            "| batch y: ",
            batch_y.numpy(),
        )



#%%
