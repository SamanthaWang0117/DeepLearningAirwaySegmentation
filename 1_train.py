import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import classes
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import Datasets
import torch.nn as nn
from torch.autograd import Variable
import metrics

from VGG16 import vggNet

path = r"C:\software\workstation\project\wss\DRIVE_2\training"
pathtest = r'C:\software\workstation\project\prac\DeepLearning\self_dataset_training\DRIVE_2\test'
train_data = Datasets(path=path)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
test_data = Datasets(path=pathtest)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)


class Trainer:
    def __init__(self, trainloader):
        self.train_loader = trainloader
        self.net = vggNet(1)
        self.net = self.net.cuda()
        self.opt = torch.optim.Adam(self.net.parameters())
        self.criterion = nn.BCELoss()
        self.criterion = self.criterion.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_list = []
        for epoch in tqdm.tqdm(range(1000)):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                # inputs, labels = Variable(inputs), Variable(labels)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.opt.step()
                running_loss += loss.item()
                if i % len(self.train_loader) == len(self.train_loader)-1:
                    print(f"[{epoch+1}, {i+1}], loss: {running_loss/len(trainloader):.3f}")
                x = inputs[0]
                # 生成的图像，取第一张
                x_ = outputs[0]
                # 标签的图像，取第一张
                y = labels[0]
                # 三张图，从第0轴拼接起来，再保存
                img = x_
                save_image(img ,os.path.join(r'./DRIVE_2/train_img',f"{i}.png"))
            loss_list.append(loss.item())

        print('Finish Training')
        torch.save(self.net, 'net.pkl')
        torch.save(self.net.state_dict(), 'net_params.pkl')
        plt.figure(1)
        plt.plot(range(len(loss_list)),np.array(loss_list))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

def reload_net():
    trainednet = torch.load('./net.pkl')
    return trainednet

class Tester:
    def __init__(self):
        self.test_loader = test_loader
        self.net = vggNet(1)
        self.net.reload_net()
        self.net = self.net.cuda()
        self.net.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for idx, batch in enumerate(self.test_loader):
            image = batch[0].float()
            label = batch[1].float()
            pred_val = self.net(image)
            acc, sen = metrics(pred_val,label,pred_val)




if __name__ == '__main__':
    t = Trainer(train_loader)
