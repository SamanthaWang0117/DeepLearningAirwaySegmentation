import os
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import Datasets
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

from VGG16 import vggNet
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
path = r"C:\software\workstation\project\wss\DRIVE_2\training"
train_data = Datasets(path=path)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)



class Trainer:
    def __init__(self, trainloader):
        self.train_loader = trainloader
        self.net = vggNet(1)
        self.net = self.net.cuda()
        self.opt = torch.optim.Adam(self.net.parameters())
        self.criterion = nn.BCELoss()
        self.criterion = self.criterion.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        for epoch in tqdm.tqdm(range(3000)):
            for i, data in enumerate(self.train_loader, 0):

                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                # inputs, labels = Variable(inputs), Variable(labels)
                # inputs, labels = inputs.to(self.device),labels.to(self.device)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.opt.step()
                if (i % 100 == 0):
                    print("i:", loss)
                x = inputs[0]

                x_ = outputs[0]

                y = labels[0]

                img = x_
                save_image(img ,os.path.join(r'C:\software\workstation\project\wss\DRIVE_2\train_img',f"{i}.png"))


        print('Finish Training')
        torch.save(self.net, 'net.pkl')
        torch.save(self.net.state_dict(), 'net_params.pkl')


if __name__ == '__main__':
    t = Trainer(train_loader)

