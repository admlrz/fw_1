import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import datasets, transforms
from datetime import datetime
import sys
from torchsummary import summary
import nuit,mymodule
import os


class Huochai_with_new_layer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.base_model = mymodule.Huochai()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = {
        "train": transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.4914, 0.4822, 0.4465))]),
        "val":transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.4914, 0.4822, 0.4465))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    data_root = os.path.join(data_root,"deep-learning-for-image-processing","MyDate","mydata")
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "all_mydata", "train"),
                                      transform=transform["train"])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=True,
                                                   batch_size=1)
    # nuit.get_mean_std(dataset=train_dataset)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_root, "all_mydata", "val"),
                                     transform=transform["val"])
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  shuffle=False,
                                                  batch_size=1)
    train_dataset_lens = len(train_dataset)
    test_dataset_lens = len(test_dataset)
    t = len(train_dataloader)
    print("batch的组数：{}".format(t))
    print(f'训练集的长度{train_dataset_lens},测试集的长度{test_dataset_lens}')

    # 定义模型架构
    net = mymodule.Huochai()

    model_weight_path = './weight/model-9.pth'
    model_weights = (torch.load(model_weight_path, map_location=device))

    net = Huochai_with_new_layer(num_classes=5)
    missing_keys, unexpected_keys = net.base_model.load_state_dict(model_weights)
    #这里只能false，新的模型已经和原来的不一样了

    for param in net.base_model.parameters():
        param.requires_grad = False

    net.to(device)

    Loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=0.001)

    TAMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    mylist = []
    for epoch in range(10):
        writer = SummaryWriter('./logs_train_DELADD/time{}/result_{}'.format(TAMESTAMP, epoch))
        print(f'----------第{epoch}轮训练开始----------')
        total_train_step = 0
        total_test_step = 0
        now_allbatch_loss = 0
        now_allbatch_test_loss = 0
        acc = 0
        net.train()
        for batch_idx, data in enumerate(train_dataloader):

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images).to(device)

            per_batch_loss = Loss_function(outputs, labels)

            optimizer.zero_grad()
            per_batch_loss.backward()
            optimizer.step()

            now_allbatch_loss += per_batch_loss

            total_train_step = total_train_step + 1
            now_avg_loss_1 = (now_allbatch_loss / total_train_step)

            if batch_idx % 5 == 4:
                print(f'训练batch次数{total_train_step}，当前训练batch的loss：{per_batch_loss:.4f}，目前总loss：'
                      f'{now_allbatch_loss:.4f},当前平均loss:{now_avg_loss_1:.4f}'
                      )
                writer.add_scalar('train_loss', per_batch_loss, total_train_step)
        net.eval()
        with torch.no_grad():
            print('--------------------测试开始----------------测试开始---------------测试开始--------------------')
            for test_batch_idx, data in enumerate(test_dataloader):
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = net(imgs).to(device)
                predict_sure = torch.max(outputs, dim=1)[1]

                now_per_test_loss = Loss_function(outputs, targets)

                now_allbatch_test_loss += now_per_test_loss

                total_test_step = total_test_step + 1

                now_avg_loss = (now_allbatch_test_loss / total_test_step)

                acc = torch.eq(predict_sure, targets).sum().item() / len(targets)
                if test_batch_idx % 1 == 0:
                    print(f'测试batch次数：{total_test_step}，当前测试batch的loss:{now_per_test_loss:.4f}'
                          f'目前的总loss：{now_allbatch_test_loss:.4f}，当前平均loss：{now_avg_loss:.4f}'
                          f'准确率:{acc}')
                    writer.add_scalar('test_loss', now_per_test_loss, total_test_step)

        mylist.append(now_avg_loss_1.item())

        if not os.path.exists('./weight_DELADD'):
            os.makedirs('./weight_DELADD')
        torch.save(net.state_dict(), "./weight_DELADD/model-{}.pth".format(epoch))

    writer = SummaryWriter('./logs_train_DELADD/time{}/result_{}'.format(TAMESTAMP,'totalloss'))
    for i in range(10):
        writer.add_scalar('epoch_loss', mylist[i], i)

    print('----------训练结束--------')
    writer.close()

if __name__ == '__main__':
    main()
