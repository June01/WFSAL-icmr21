'''This script defines for training in episode in Few-shot learning setting:
    1. VideoDataset: extract dataset annotation, features and so on.
    2. ClassBalancedSampler: get sample set
    3. BatchData: get batch data
'''
import random
import json
import torch
import os.path as osp
import numpy as np

import torch.utils.data as data
from .collation import collate_fn_padd
from .util import strlist2multihot
from functools import partial

class VideoDataset(object):

    """Dataset class"""
    def __init__(self, dataset, config, shuffle=True, split=None):

        self.shuffle = shuffle
        # Initialize all the arguments
        # print(">>> Loading features and annotations from {}".format(dataset))
        anno_dict = json.load(open(config.anno_dict_path))
        self.annos = anno_dict['database']
        # TODO: Feature can also be used as two-stream and train separately
        self.feats = np.load(config.feat_path, encoding='bytes', allow_pickle=True)
        classlist = np.load(osp.join(config.root, '{}-Annotations/classlist.npy'.format(dataset)), allow_pickle=True)
        self.classlist = [clx.decode('utf-8') for clx in classlist]

        self.video_idxs = config.video_idxs

        self.activity_net = False
        if dataset == 'ActivityNet1.2':
            self.activity_net = True
        # print(">>> Prepare test videos")
        subset_train = 'training' if self.activity_net else 'validation'
        subset_test = 'validation'  if self.activity_net else 'testing'
        self.groups_seg, self.groups_video = self.group_video_by_cat(subset=subset_train)
        self.train_cls, self.test_cls = self.split_dataset(split)
        # Prepare whole test data
        self.test_set = []
        self.test_labels = {}
        for vid in self.video_idxs:
            if vid == 'video_test_0000270' or vid == 'video_test_0001496':
                continue
            if self.annos[vid]['subset'] == subset_test:
                label_set = []
                for anno in self.annos[vid]['annotations']:
                    label = anno['label']
                    label_set.append(label)
                if len(set(label_set).intersection(self.test_cls)) > 0:
                    self.test_set.append(vid)
                    self.test_labels[vid] = set(label_set)

        if self.shuffle:
            random.shuffle(self.test_set)

    # Classwise video grouping
    def group_video_by_cat(self, subset='training'):
        groups_seg = dict()
        groups_video = dict()
        for cls in self.classlist:
            groups_seg[cls] = {}
            groups_video[cls] = []

        for vid in self.annos.keys():
            ss = self.annos[vid]['subset']
            if ss == subset:
                for anno in self.annos[vid]['annotations']:
                    label = anno['label']
                    if vid not in groups_video[label]:
                        groups_video[label].append(vid)
                        groups_seg[label][vid] = []
                    groups_seg[label][vid].append(anno['segment'])

        if self.shuffle:
            for cls in self.classlist:
                random.shuffle(groups_video[cls])

        return groups_seg, groups_video

    def split_dataset(self, split=None):
        if split is None:
            train_cls = self.classlist
            test_cls = self.classlist
        elif split=='cvpr18':
            if self.activity_net:
                train_cls = self.classlist[:80]
                test_cls = self.classlist[80:]
            else:
                train_cls = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving',
                             'CricketBowling']
                test_cls = ['CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
                        'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing',
                        'ThrowDiscus', 'VolleyballSpiking']
        elif split == 'test':
            if self.activity_net:
                train_cls = self.classlist[:86]
                test_cls = self.classlist[86:]
            else:
                train_cls = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving',
                             'CricketBowling', 'CricketShot', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'Shotput',
                             'SoccerPenalty', 'LongJump', 'TennisSwing', 'ThrowDiscus']
                test_cls = ['Diving', 'HighJump', 'JavelinThrow', 'PoleVault', 'VolleyballSpiking']
        else:
            if self.activity_net:
                train_cls = self.classlist[86:]
                test_cls = self.classlist[:86]
            else:
                test_cls = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving',
                             'CricketBowling']
                train_cls = ['CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
                        'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing',
                        'ThrowDiscus', 'VolleyballSpiking']

        return train_cls, test_cls

    def pick_test_vid(self, pick_classes, test_vids):
        '''Pick test videos in picked classes
        '''
        test_pick_vids = []
        test_pick_labels = {}
        for vid in test_vids:
            label_set = []
            for anno in self.annos[vid]['annotations']:
                label = anno['label']
                label_set.append(label)
            if len(set(label_set).intersection(pick_classes)) > 0:
                test_pick_vids.append(vid)
                test_pick_labels[vid] = set(label_set)
        return test_pick_vids, test_pick_labels

    def pick_class_ep(self, train_cls, pick_num=5, sample_num=1, train_num=6, mode='training'):
        # pick_num: number of picked classes
        # sample_num: number of samples
        # train_num: number of training samples

        pick_classes = random.sample(train_cls, pick_num)
        if self.shuffle:
            random.shuffle(pick_classes)
        train_vids = []; train_labels = []
        sample_vids = []; sample_labels = []
        for i in range(pick_num):
            k = pick_classes[i]
            random.shuffle(self.groups_video[k])
            for vid in self.groups_video[k][:sample_num]:
                sample_vids.append(vid)
                sample_labels.append(i)
                # print('sample', vid, k, i+1)
            for vid in self.groups_video[k][sample_num:sample_num+train_num]:
                train_vids.append(vid)
                train_labels.append(i)

        # We don't shuffle here to facilitate usage of classbalancedsampler
        return pick_classes, train_vids, train_labels, sample_vids, sample_labels

class ClassBalancedSampler(data.Sampler):
    '''
    Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class'
    '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        # print(batch)
        return iter(batch)

    def __len__(self):
        return 1

class BatchData(data.Dataset):

    # split: ['train', 'sample']
    def __init__(self, videodb, video_ids, labels, picked_class, split='train', shuffle=True):

        self.shuffle = shuffle
        self.split = split
        self.videodb = videodb
        self.video_ids = video_ids
        if self.split == 'sample':
            self.groups_seg = self.videodb.groups_seg

        # A list of classes, e.g.['BasketballDunk', 'HighJump']
        self.picked_class = picked_class
        self.labels = labels
        # self.activitynet = activitynet

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        feat = self.videodb.feats[self.videodb.video_idxs.index(vid)]
        if self.labels == None:
            label = 0
        else:
            label = self.labels[idx]

        if self.split == 'sample':
            cls = self.picked_class[label]
            segment = random.choice(self.groups_seg[cls][vid])
            start = int(segment[0] * 25 / 16.0)
            end = int(segment[1] * 25 / 16.0)
            feat = feat[start:end + 1]
        return feat.shape[0], feat, label, idx

    def __len__(self):
        return len(self.video_ids)

def get_data_loader(videodb, picked_class, fetch_num, idx, labels, num_per_class=1, split='train', shuffle = True):
    pick_num = len(picked_class)

    dataset = BatchData(videodb, idx, labels, picked_class, split=split, shuffle=True)

    sampler = ClassBalancedSampler(num_per_class, pick_num, fetch_num, shuffle)
    if split == 'test':
        loader = data.DataLoader(dataset, batch_size=fetch_num, collate_fn= partial(collate_fn_padd, max_len=0))
    else:
        loader = data.DataLoader(dataset, batch_size=num_per_class*pick_num, sampler=sampler, collate_fn = partial(collate_fn_padd, max_len=0))

    return loader