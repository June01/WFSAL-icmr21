import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
from .linearsq import LinearSQ

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=81, feat_dim=128, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, feature, labels):
        """
        Args:
            feature: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = feature.size(0)
        distmat = torch.pow(feature, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # import pdb
        # pdb.set_trace()
        distmat.addmm_(1, -2, feature, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()

        if labels.numel() > labels.size(0):
            mask = labels > 0
        else:
            labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = labels.eq(classes.expand(batch_size, self.num_classes).float())

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value *= labels[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

class FCEncoder(nn.Module):
    def __init__(self, n_feature=2048, out_dim=128):
        super(FCEncoder, self).__init__()
        self.n_featureby2 = int(n_feature / 2)
        # FC layers for the 2 streams
        self.fc_f = nn.Linear(self.n_featureby2, self.n_featureby2)
        self.fc1_f = nn.Linear(self.n_featureby2, out_dim)
        self.fc_r = nn.Linear(self.n_featureby2, self.n_featureby2)
        self.fc1_r = nn.Linear(self.n_featureby2, out_dim)
        self.relu = nn.ReLU(True)
        self.dropout_f = nn.Dropout(0.5)
        self.dropout_r = nn.Dropout(0.5)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        # inputs - batch x seq_len x featSize
        base_x_f = inputs[:, :, self.n_featureby2:]
        base_x_r = inputs[:, :, :self.n_featureby2]
        x_f = self.relu(self.fc_f(base_x_f))
        x_r = self.relu(self.fc_r(base_x_r))
        if is_training:
            x_f = self.dropout_f(x_f)
            x_r = self.dropout_r(x_r)
        x_f = self.relu(self.fc1_f(x_f))
        x_r = self.relu(self.fc1_r(x_r))
        enc = torch.cat((x_f, x_r), -1)
        return enc

class Classifier(nn.Module):
    def __init__(self, n_feature=256, num_class=81):
        super(Classifier, self).__init__()
        self.n_featureby2 = int(n_feature/2)
        self.fc_f = nn.Linear(self.n_featureby2, num_class)
        self.fc_r = nn.Linear(self.n_featureby2, num_class)
        self.apply(weights_init)
        self.mul_r = nn.Parameter(data=torch.Tensor(num_class).float().fill_(1))
        self.mul_f = nn.Parameter(data=torch.Tensor(num_class).float().fill_(1))

    def forward(self, inputs):

        base_x_f = inputs[:, :, :self.n_featureby2]
        base_x_r = inputs[:, :, self.n_featureby2:]

        cls_x_f = self.fc_f(base_x_f)
        cls_x_r = self.fc_r(base_x_r)

        tcam = cls_x_r * self.mul_r + cls_x_f * self.mul_f
        return cls_x_f, cls_x_r, tcam

class AttentionGenerator(nn.Module):
    def __init__(self, in_dim=4, bn_flag=True):
        super(AttentionGenerator, self).__init__()

        # self.k = k
        # self.conv = nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=int((k-1)/2), dilation=1, bias=True)
        # self.sigmoid = nn.Sigmoid()
        self.bn_flag = bn_flag
        # self.fc1 = nn.Linear(in_dim, 4)
        # self.fc2 = nn.Linear(4, 1)
        # self.relu = nn.ReLU(True)
        self.fc = LinearSQ(in_dim, 1)
        if self.bn_flag:
            # print('It is using batch normalization of 2')
            # self.bn1 = nn.BatchNorm1d(in_dim)
            # self.bn2 = nn.BatchNorm1d(4)
            self.bn = nn.BatchNorm1d(in_dim)

        self.apply(weights_init)

    def forward(self, cmp):
        '''
        :param cmp: [bs, num_class*sample_per_class, length_query, length_sample, d]
        :return: tsm: [bs, num_class*sample_per_class, length_query, length_sample]
        '''
        if self.bn_flag:
            b, s, l, d = cmp.size()
            cmp = cmp.view(-1, l, d)
            # cmp = F.sigmoid(cmp)
            # relu
            # mask = self.relu(self.fc1(self.bn(cmp.transpose(1,2)).transpose(1,2)))
            # mask = self.fc1(self.bn1(cmp.transpose(1,2)).transpose(1,2))
            # mask = self.fc2(self.bn2(mask.transpose(1,2)).transpose(1,2))
            mask = self.fc(self.bn(cmp.transpose(1, 2)).transpose(1, 2))
            mask = mask.squeeze(2).view(b,s,l)
        else:
            mask = self.fc(cmp).squeeze(-1)

        return mask

