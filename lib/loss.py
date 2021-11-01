import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

def get_loss(features, logits, labels, seq_len, centers, num_classes, batch_mean=[], mean_flag=False, distance='cosine', device=torch.device("cuda")):
    ''' features: torch tensor dimension (B, n_element, feature_size),
        logits: torch tensor of dimension (B, n_element, n_class),
        labels: torch tensor of dimension (B, n_class) one-hot labels,
        centers: (num_class, feature_size)
        seq_len: numpy array of dimension (B,) indicating the length of each video in the batch,
        criterion: center loss criterion,
        return: torch tensor of dimension 0 (value) '''

    if mean_flag:
        distmat = get_distmat_mean(batch_mean, centers, len(centers))
    else:
        distmat, _ = get_distmat(features, logits, seq_len, centers, num_classes, distance, device=torch.device("cuda"))
    loss_cls = clsloss(distmat, labels)

    return loss_cls

def entropyloss(logits):
    logits_sgm = torch.sigmoid(logits)
    norm = torch.norm(logits_sgm, p=1, dim=1, keepdim=True)
    logits_l1 = logits_sgm / norm
    m = (logits_l1 > 1e-5) * logits_l1 + (logits_l1 < 1e-5) + 1e-5
    entropy = -m * torch.log(m)
    return torch.mean(torch.sum(entropy, dim=1))

def clsloss(distmat, labels):
    # calculate cls loss
    log_sm = F.log_softmax(distmat, dim=1)
    if len(labels) != len(log_sm):
        # print(labels.size(), log_sm.size(), feature.size(), labels.size())
        assert len(labels) == len(log_sm)

    loss_cls = -torch.mean(torch.sum(Variable(labels) * log_sm, dim=1), dim=0)
    return loss_cls

def get_distmat_mean(batch_mean, mean_feats, num_class):

    distmat = -torch.sum(torch.pow(batch_mean.unsqueeze(1).repeat(1, num_class, 1)
                                       - mean_feats.unsqueeze(0), 2), dim=-1)
    return distmat

def get_distmat(features, logits, seq_len, centers, num_classes, distance='cosine', device=torch.device("cuda")):
    '''
    features: torch tensor dimension (B, n_element, feature_size),
    logits: torch tensor of dimension (B, n_element, n_class),
    labels: torch tensor of dimension (B, n_class) one-hot labels,
    seq_len: numpy array of dimension (B,) indicating the length of each video in the batch,
    criterion: center loss criterion,
    return: torch tensor of dimension 0 (value)
    '''


    batch_size = len(features)
    num_samples = len(centers)
    num_dim = centers.size(-1)

    feat = torch.zeros(0).to(device)
    for i in range(features.size(0)):

        atn = F.softmax(logits[i][:seq_len[i]], dim=0)
        # aggregate features category-wise
        for l in range(num_samples):
            atnl = atn[:, l]
            atnl = atnl.unsqueeze(1).expand(seq_len[i], features.size(2))
            # attention-weighted feature aggregation
            featl = torch.sum(features[i][:seq_len[i]] * atnl, dim=0, keepdim=True)
            feat = torch.cat([feat, featl], dim=0)

    feat = feat.view(-1, num_samples, num_dim)

    if distance == 'eu':
        distmat = -torch.sum(torch.pow(feat - centers.unsqueeze(0).expand(batch_size, num_samples, num_dim), 2),
                             dim=-1)
    elif distance == 'l2':
        distmat = -torch.sqrt(torch.sum(torch.pow(feat - centers.unsqueeze(0).expand(batch_size, num_samples,
                                                                                        num_dim), 2), dim=-1))
    elif distance == 'cosine':
        distmat = torch.nn.functional.cosine_similarity(feat,
                                                        centers.unsqueeze(0).expand(batch_size, num_samples, num_dim),
                                                        dim=-1,
                                                        eps=1e-8)

    # distmat [batch_num_per_class*class_num, class_num*sample_num_per_class]
    distmat = distmat.view(batch_size, num_classes, -1)
    distmat = torch.mean(distmat, axis=-1)

    return distmat, feat
