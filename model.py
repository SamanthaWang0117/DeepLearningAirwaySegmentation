import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class vggNet(nn.Module):
    def __init__(self, num_classes):
        super(vggNet, self).__init__()
        self.branch_1_1 = nn.Conv2d(1, 64, (3, 3), padding=(1, 1))
        self.branch_1_2 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))

        self.branch_2_1 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.branch_2_2 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1))

        self.branch_3_1 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.branch_3_2 = nn.Conv2d(256, 256, (3, 3), padding=(1, 1))
        self.branch_3_3 = nn.Conv2d(256, 256, (3, 3), padding=(1, 1))

        self.branch_4_1 = nn.Conv2d(256, 512, (3, 3), padding=(1, 1))
        self.branch_4_2 = nn.Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.branch_4_3 = nn.Conv2d(512, 512, (3, 3), padding=(1, 1))

        self.pooling = nn.MaxPool2d(2, stride=2)
        num_ch = 16

        self.spe_1 = torch.nn.Conv2d(64, num_ch, (3, 3), padding=(1, 1))
        self.spe_2 = torch.nn.Conv2d(128, num_ch, (3, 3), padding=(1, 1))
        self.spe_3 = torch.nn.Conv2d(256, num_ch, (3, 3), padding=(1, 1))
        self.spe_4 = torch.nn.Conv2d(512, num_ch, (3, 3), padding=(1, 1))
        self.upsamp_2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.upsamp_4 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        self.upsamp_8 = torch.nn.Upsample(scale_factor=8, mode='nearest')

        self.output = torch.nn.Conv2d(num_ch * 4, num_classes, (1, 1), padding=(0, 0))

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

        part_2_1 = self.spe_2(branch2)

        part_2 = self.upsamp_2(part_2_1)

        branch3 = self.pooling(branch2)
        branch3 = self.branch_3_1(branch3)
        branch3 = F.relu(branch3)
        branch3 = self.branch_3_2(branch3)
        branch3 = F.relu(branch3)
        branch3 = self.branch_3_2(branch3)
        branch3 = F.relu(branch3)

        part_3_1 = self.spe_3(branch3)
        part_3 = self.upsamp_4(part_3_1)

        branch4 = self.pooling(branch3)
        branch4 = self.branch_4_1(branch4)
        branch4 = F.relu(branch4)
        branch4 = self.branch_4_2(branch4)
        branch4 = F.relu(branch4)
        branch4 = self.branch_4_3(branch4)
        branch4 = F.relu(branch4)

        part_4_1 = self.spe_4(branch4)
        part_4 = self.upsamp_8(part_4_1)

        tensor_set = [part_1, part_2, part_3, part_4]
        tensor_drop = [part_1, part_2_1, part_3_1, part_4_1]

        spe_concat = torch.cat(tensor_set, dim=1)
        spe_concat_final = self.output(spe_concat)

        #output_final = torch.sigmoid(spe_concat_final)
        output_final = spe_concat_final

        return output_final, tensor_drop , spe_concat


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input.float(), self.W.float())  # shape [N, out_features]
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        out_feats = F.elu(h_prime)
        out_prob = torch.sigmoid(out_feats.mean(dim=1)).reshape((1,int(np.sqrt(adj.shape[-1])), int(np.sqrt(adj.shape[-1]))))
        return out_feats, out_prob

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'



class gat(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(gat, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class infer_module(nn.Module):
    def __init__(self, nfeat, dropout, alpha, device, base_dir):
        super(infer_module, self).__init__()

        self.nfeat = nfeat
        self.branch_inf_1 = nn.Conv2d(32, 16, (1, 1), stride=(1, 1))
        self.branch_inf_2 = nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.branch_inf_3 = nn.Conv2d(32, 16, kernel_size=(1,1))
        self.upsamp_1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(dropout)

        self.output_final = nn.Conv2d(16 * 2, 1, (1, 1), padding=(0, 0))

    def forward(self, cnn_drop,x):
        cnn1 = cnn_drop[0]
        cnn2 = cnn_drop[1]
        cnn3 = cnn_drop[2]
        cnn4 = cnn_drop[3]

        branch_1 = self.branch_inf_1(x)
        branch_1 = F.relu(branch_1)
        branch_1 = self.upsamp_1(branch_1)
        cnn1_drop = self.dropout(cnn4)
        part1_set = [cnn1_drop, branch_1]
        part1 = torch.cat(part1_set, dim=1)


        branch_2 = self.branch_inf_2(part1)
        branch_2 = F.relu(branch_2)
        branch_2 = self.upsamp_1(branch_2)
        cnn2_drop = self.dropout(cnn3)
        part2_set = [cnn2_drop, branch_2]
        part2 = torch.cat(part2_set, dim=1)


        branch_3 = self.branch_inf_2(part2)
        branch_3 = F.relu(branch_3)
        branch_3 = self.upsamp_1(branch_3)
        cnn3_drop = self.dropout(cnn2)
        part3_set = [cnn3_drop, branch_3]
        part3 = torch.cat(part3_set, dim=1)


        branch_4 = self.branch_inf_2(part3)
        branch_4 = F.relu(branch_4)
        branch_4 = self.upsamp_1(branch_4)
        cnn4_drop = self.dropout(cnn1)
        part4_set = [cnn4_drop, branch_4]
        part4 = torch.cat(part4_set, dim=1)

        part_4_output = self.output_final(part4)

        output_final = part_4_output

        return output_final