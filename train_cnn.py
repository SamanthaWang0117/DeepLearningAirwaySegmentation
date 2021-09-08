import os
from copy import copy, deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from load_air_data import Datasets
import torch.nn as nn
from torch.autograd import Variable

from model import vggNet

base_dir = '/Users/samanthawang/Downloads/segmentationvgn'
# path = r"/Users/samanthawang/Documents/segmentationvgn/DRIVE_2/training/"
train_path = r'./DRIVE_2/training'
test_path = r'./DRIVE_2/test'

train_data = Datasets(path=train_path)
train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)
test_data = Datasets(path=test_path)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)


def dice_coef(y, y_pred):
    gt = np.ndarray.flatten(copy(y))
    pred = np.ndarray.flatten((copy(y_pred)))
    return (2 * np.sum(gt * pred)) / (np.sum(gt) + np.sum(pred))


class Trainer:
    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.net = vggNet(1).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        train_loss_list = []
        train_dice_list = []
        test_loss_list = []
        test_dice_list = []
        best_train_loss = 100.
        best_train_dice = 0.
        best_test_loss = 100.
        best_test_dice = 0.

        epochs = 300
        for epoch in range(epochs):
            print("Training epoch %i out of %i" % (epoch, epochs))
            running_loss = 0.0
            running_dice = 0.0
            self.net = self.net.train()
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data['img'], data['label']
                # inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs[0], labels)
                loss.backward()
                self.opt.step()
                running_loss += loss.item()

                x = torch.nn.Sigmoid()(outputs[0])
                y_pred_np = x.detach().cpu().numpy()
                y_pred_np[y_pred_np >= 0.5] = 1.0
                y_pred_np[y_pred_np < 0.5] = 0.0
                labels = labels.detach().cpu().numpy()
                dice = dice_coef(labels, y_pred_np)
                running_dice += dice

                img = x
                if (i % 20) == 0:
                    save_image(img, os.path.join(r'./DRIVE_2/train_img', f"train_ep{epoch}_{i}.png"),
                        normalize=True)

            mean_loss = running_loss / len(self.train_loader)
            mean_dice = running_dice / len(self.train_loader)
            print(f"[{epoch + 1}, {i + 1}], training loss: {mean_loss:.4f}")
            print(f"[{epoch + 1}, {i + 1}], training Dice: {mean_dice:.4f}")

            train_loss_list.append(mean_loss)
            train_dice_list.append(mean_dice)
            if mean_loss < best_train_loss:
                best_train_loss = mean_loss
                torch.save(deepcopy(self.net.state_dict()), r'./model_trainloss.pth')
            if mean_dice > best_train_dice:
                best_train_dice = mean_dice
                torch.save(deepcopy(self.net.state_dict()), r'./model_traindice.pth')

            with torch.no_grad():
                self.net = self.net.eval()
                test_loss = []
                test_dice = []
                for i, input_dict in enumerate(self.test_loader):
                    images = input_dict['img']
                    labels = input_dict['label']
                    images = images.type(torch.FloatTensor).to(self.device)
                    labels = labels.type(torch.FloatTensor).to(self.device)

                    prob_map, _, _ = self.net(images)
                    loss = self.criterion(prob_map, labels)
                    test_loss.append(loss.item())

                    prob_map_thresh = torch.nn.Sigmoid()(prob_map).detach().cpu().numpy()
                    prob_map_thresh[prob_map_thresh < 0.5] = 0.
                    prob_map_thresh[prob_map_thresh >= 0.5] = 1.
                    tmp_test_dice = dice_coef(labels.detach().cpu().numpy(), prob_map_thresh)
                    test_dice.append(tmp_test_dice)

                    y_pred = torch.nn.Sigmoid()(prob_map)
                    y_pred_np = y_pred.detach().cpu().numpy().astype(np.float32)
                    y_pred_np_thresh = y_pred_np
                    y_pred_np_thresh[y_pred_np >= 0.5] = 1.0
                    y_pred_np_thresh[y_pred_np < 0.5] = 0.0
                    if (i % 20) == 0:
                        save_image(y_pred.detach().cpu(),
                                   os.path.join(r'./DRIVE_2/test_img', f"test_ep{epoch}_{i}_pred.png"),
                                   normalize=True)
                        save_image(torch.from_numpy(y_pred_np_thresh),
                                   os.path.join(r'./DRIVE_2/test_img', f"test_ep{epoch}_{i}_pred_thresh.png"),
                                   normalize=True)
                        save_image(labels.detach().cpu(),
                                   os.path.join(r'./DRIVE_2/test_img', f"test_ep{epoch}_{i}_gt.png"),
                                   normalize=True)
                        save_image(images.detach().cpu(),
                                   os.path.join(r'./DRIVE_2/test_img', f"test_ep{epoch}_{i}_im.png"),
                                   normalize=True)

                mean_test_loss = np.mean(test_loss)

                test_loss_list.append(mean_test_loss)
                if mean_test_loss < best_test_loss:
                    best_test_loss = mean_test_loss
                    torch.save(deepcopy(self.net.state_dict()), r'./best_model_testloss.pth')
                mean_test_dice = np.mean(test_dice)
                test_dice_list.append(mean_test_dice)
                if mean_test_dice > best_test_dice:
                    best_test_dice = mean_test_dice
                    torch.save(deepcopy(self.net.state_dict()), r'./best_model_testdice.pth')
            print(f"[{epoch + 1}, {i + 1}], test loss: {mean_test_loss:.4f}")
            print(f"[{epoch + 1}, {i + 1}], test Dice: {mean_test_dice:.4f}")

            plt.figure(1)
            plt.plot(range(len(train_loss_list)), np.array(train_loss_list))
            plt.plot(range(len(test_loss_list)), np.array(test_loss_list))
            plt.legend(["Training", "Testing"])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(r'./DRIVE_2', "Loss_vs_epochs_graph.png"))
            plt.close()

            plt.figure(2)
            plt.plot(range(len(train_dice_list)), np.array(train_dice_list))
            plt.plot(range(len(test_dice_list)), np.array(test_dice_list))
            plt.legend(["Training", "Testing"])
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.ylim((0,1))
            plt.savefig(os.path.join(r'./DRIVE_2', "Dice_vs_epochs_graph.png"))
            plt.close()

            # Save mean training/test loss/dice to .txt file
            with open(r'./Train_Loss.txt', 'w') as f:
                for item in train_loss_list:
                    f.write("%f\n" % item)
            with open(r'./Train_Dice.txt', 'w') as f:
                for item in train_dice_list:
                    f.write("%f\n" % item)
            with open(r'./Test_Loss.txt', 'w') as f:
                for item in test_loss_list:
                    f.write("%f\n" % item)
            with open(r'./Test_Dice.txt', 'w') as f:
                for item in test_dice_list:
                    f.write("%f\n" % item)

        print('Finish training and validating')
        print('Best training loss: ')
        print(str(best_train_loss))
        print('Best training Dice: ')
        print(str(best_train_dice))
        print('Best test loss: ')
        print(str(best_test_loss))
        print('Best test Dice: ')
        print(str(best_test_dice))
        # torch.save(self.net, 'net.pkl')
        # torch.save(self.net.state_dict(), 'net_params.pkl')


if __name__ == '__main__':
    t = Trainer(train_loader, test_loader)
