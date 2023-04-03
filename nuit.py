import torch
from tqdm import tqdm
import train_my

def get_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,batch_size=1, shuffle=True,num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print(mean[0])
    print(mean[1])
    print(mean[2])
    for inputs, targets in tqdm(dataloader):
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

if __name__ == '__get_mean_std__':
    get_mean_std(train.train_dataset)