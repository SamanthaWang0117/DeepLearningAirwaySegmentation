import torch
import torch.nn.functional as F


class vggNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(vggNet, self).__init__()
        self.branch_1_1 = torch.nn.Conv2d(3, 64, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_1_2 = torch.nn.Conv2d(64, 64, (3, 3),stride=(1,1),padding=(1,1))

        self.branch_2_1 = torch.nn.Conv2d(64, 128, (3, 3),stride=(1,1), padding=(1,1))
        self.branch_2_2 = torch.nn.Conv2d(128, 128, (3, 3),stride=(1,1),padding=(1,1))

        self.branch_3_1 = torch.nn.Conv2d(128, 256, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_3_2 = torch.nn.Conv2d(256, 256, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_3_3 = torch.nn.Conv2d(256, 256, (3, 3),stride=(1,1),padding=(1,1))

        self.branch_4_1 = torch.nn.Conv2d(256, 512, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_4_2 = torch.nn.Conv2d(512, 512, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_4_3 = torch.nn.Conv2d(512, 512, (3, 3),stride=(1,1),padding=(1,1))

        self.pooling = torch.nn.MaxPool2d(2, stride=2)
        num_ch = 16

        self.spe_1 = torch.nn.Conv2d(64, num_ch, (3, 3),stride=(1,1),padding=(1,1))
        self.spe_2 = torch.nn.Conv2d(128, num_ch, (3, 3),stride=(1,1),padding=(1,1))
        self.spe_3 = torch.nn.Conv2d(256, num_ch, (3, 3),stride=(1,1),padding=(1,1))
        self.spe_4 = torch.nn.Conv2d(512, num_ch, (3, 3),stride=(1,1),padding=(1,1))
        self.upsamp_2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.upsamp_4 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        self.upsamp_8 = torch.nn.Upsample(scale_factor=8, mode='nearest')

        self.output = torch.nn.Conv2d(num_ch * 4, num_classes, (1, 1),padding=(0, 0))

    def forward(self, x):
        
        branch1 = self.branch_1_1(x)
        branch1 = F.relu(branch1)
        branch1 = self.branch_1_2(branch1)
        branch1 = F.relu(branch1)

        part_1 = self.spe_1(branch1)

        branch2 = self.pooling(branch1)
        branch2 = self.branch_2_1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.branch_2_2(branch2)
        branch2 = F.relu(branch2)

        part_2 = self.spe_2(branch2)
        part_2 = self.upsamp_2(part_2)

        branch3 = self.pooling(branch2)
        branch3 = self.branch_3_1(branch3)
        branch3 = F.relu(branch3)
        branch3 = self.branch_3_2(branch3)
        branch3 = F.relu(branch3)
        branch3 = self.branch_3_2(branch3)
        branch3 = F.relu(branch3)

        part_3 = self.spe_3(branch3)
        part_3 = self.upsamp_4(part_3)

        branch4 = self.pooling(branch3)
        branch4 = self.branch_4_1(branch4)
        branch4 = F.relu(branch4)
        branch4 = self.branch_4_2(branch4)
        branch4 = F.relu(branch4)
        branch4 = self.branch_4_3(branch4)
        branch4 = F.relu(branch4)

        part_4 = self.spe_4(branch4)
        part_4 = self.upsamp_8(part_4)

        tensor_set = [part_1, part_2, part_3, part_4]
        spe_concat = torch.cat(tensor_set, dim=1)
        spe_concat_final = self.output(spe_concat)

        output_final = F.sigmoid(spe_concat_final)

        return output_final

if __name__ == '__main__':
    a = torch.randn(1,3, 256, 256)
    net = Net(num_classes=1)
    print(net(a).shape)