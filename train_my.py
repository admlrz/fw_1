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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # vgg = models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    # in_features = vgg.classifier[6].in_features
    # vgg.fc = nn.Linear(in_features, 10)

    # train_dataset = datasets.CIFAR10(root='./data', download=False,
    #                                  train=True, transform=transforms.ToTensor()
    #                                  )
    #
    # test_dataset = datasets.CIFAR10(root='./data', download=False,
    #                                 train=False, transform=transforms.ToTensor())
    # mean,std = nuit.get_mean_std(train_dataset)
    # print(mean,std)

    transform = torchvision.transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.CenterCrop(224),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.4914, 0.4822, 0.4465))
        ]
    )

    train_dataset = datasets.CIFAR10(root=".\data", train=True,
                                     download=False, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=True,
                                                   batch_size=64)
    # nuit.get_mean_std(dataset=train_dataset)
    test_dataset = datasets.CIFAR10(root=".\data", train=False,
                                    download=False, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  shuffle=False,
                                                  batch_size=64)
    train_dataset_lens = len(train_dataset)
    test_dataset_lens = len(test_dataset)
    t = len(train_dataloader)
    print("batch的组数：{}".format(t))
    print(f'训练集的长度{train_dataset_lens},测试集的长度{test_dataset_lens}')

    my = mymodule.Huochai(num_classes=10)
    net = my.to(device)
    summary(net, (3, 224, 224), batch_size=64)

    # images, labels = next(iter(test_dataloader))
    # print(train_dataset.data.shape)
    # print(images.shape)

    Loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    TAMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    mylist = []
    for epoch in range(10):
        writer = SummaryWriter('./logs_mytrain/time{}/result_{}'.format(TAMESTAMP, epoch))
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

            # print(f'训练batch次数{total_train_step}，当前训练batch的loss：{per_batch_loss}，目前总loss：'
            #       f'{now_allbatch_loss},当前平均loss:{now_avg_loss}'
            #       )

            if batch_idx % 100 == 99:
                print(f'训练batch次数{total_train_step}，当前训练batch的loss：{per_batch_loss}，目前总loss：'
                      f'{now_allbatch_loss},当前平均loss:{now_avg_loss_1}'
                      )
                writer.add_scalar('train_loss', per_batch_loss, total_train_step)
        #writer.add_scalar('epoch_loss_1', now_avg_loss_1, epoch)

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
                if test_batch_idx % 10 == 9:
                    print(f'测试batch次数：{total_test_step}，当前测试batch的loss:{now_per_test_loss}'
                          f'目前的总loss：{now_allbatch_test_loss}，当前平均loss：{now_avg_loss}'
                          f'准确率:{acc}')
                    writer.add_scalar('test_loss', now_per_test_loss, total_test_step)

        mylist.append(now_avg_loss_1.item())
        #
        # #writer.add_scalar('epoch_loss_1', mylist[-1], epoch)

        if not os.path.exists('./weight'):
            os.makedirs('./weight')
        torch.save(net.state_dict(), "./weight/model-{}.pth".format(epoch))
    writer = SummaryWriter('./logs_mytrain/time{}/result_{}'.format(TAMESTAMP,'totalloss'))
    for i in range(10):
        writer.add_scalar('epoch_loss', mylist[i], i)


    print('----------训练结束--------')
    writer.close()


if __name__ == '__main__':
    main()
