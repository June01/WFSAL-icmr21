import sys
sys.path.append('..')
import os
import random

import json
import pdb
import time

import numpy as np
import util.utils as utils
import util.ssm as ssm
from scipy.spatial.distance import cosine
from scipy.special import softmax

from eval.get_detection_performance import eval_mAP
from eval.get_classification_performance import eval_acc

from scipy.spatial.distance import cosine
from dataset.dataset import VideoDataset
from configs.config import Config

def get_sample_set(feats, video_idxs, sample_idx, groups, picked_class, sample_num=1):
    '''This is used to get the sample set of each class during each testing epoch
    '''
    sample_set = {}
    # Random choose one video from a class and cut the corresponding action instance of class c
    for i in range(len(sample_idx)):
        vid = sample_idx[i]
        k = picked_class[i]
        sample_set[k] = []
        count = 0
        while True:
            segment = random.choice(groups[k][vid])
            vfeat = feats[video_idxs.index(vid)]
            fps = 25.0
            start = int(segment[0]*fps/16.0)
            end = int(segment[1]*fps/16.0)
            sample_set[k].append(vfeat[start:end+1])
            if end-start+1>1:
                count+=1
            if count ==sample_num:
                break
    return sample_set

def eval(feats, video_idxs, sample_set, dataset, picked_class, test_pick_vids, test_pick_labels, eval_one=False):
    '''This is used to generate attention masks from pretrained features, predict actions and evaluate action recognition and localiation performance.
    '''

    outjson = {}
    eval_class_one = []
    eval_vid = []
    print(">>> randomly choose a query video")
    count = random.choice(range(len(test_pick_vids)))
    vid = test_pick_vids[count]
    ftest = feats[video_idxs.index(vid)]
    annos = []
    scores_set = {}
    sim_dict = {}

    print(">>> Calculating frame-level attention mask")
    # Get classification score and class
    for key in sample_set:
        scores_list = []
        sim_list = []
        for fsample in sample_set[key]:
            # get mean features of 'sample_num' samples
            fsample_mean = np.mean(fsample, axis=0)
            fsample_mean_ = np.mean(fsample, axis=0, keepdims=True)
            # temporal similarity matrix calculation
            tsm = ssm.get_ssm_ip(fsample_mean_, ftest)
            tsm = np.expand_dims(tsm, axis=0)
            # attention mask calculation
            scores = np.max(tsm, axis=0)
            scores_list.append(scores)
            scores = softmax(scores)

            scorestile = np.repeat(scores[...,np.newaxis], 2048 , -1)
            overall_feat = np.sum(scorestile * ftest, axis=0)
            # overall_feat = np.mean(ftest, axis=0)
            # euclidean distance to calculate the distance between sample feature and query feature to decide action category
            sim = np.sum(np.abs(overall_feat - fsample_mean) ** 2)
            sim_list.append(sim)

        sim_dict[key] = np.mean(sim_list)
        scores_set[key]=np.mean(np.array(scores_list), axis=0)

    sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1])}
    clsx = list(sim_dict.keys())

    for i in range(1):
        key = clsx[i]

        segments = utils.postprocess(scores_set[key], activityNet=False, th=0)
        annos.extend(utils.result2json(segments, key))

    outjson[vid] = annos

    t1 = 0; t3 = 0
    if eval_one:
        eval_vid.append(vid)
        for c in test_pick_labels[vid]:
            if c in picked_class:
                eval_class_one.append(c)
        print(eval_class_one, eval_vid)
        if clsx[0] in eval_class_one:
            t1 = 1
            t3 = 1
        else:
            if (clsx[1] in eval_class_one) or (clsx[2] in eval_class_one):
                t3 = 1

    final_result = dict()
    final_result['version'] = 'VERSION 1.2'
    final_result['external_data'] = []
    final_result['results'] = outjson

    outpath = 'output_no_learning_{}_eutsm.json'.format(dataset)
    with open(outpath, 'w') as fp:
        json.dump(final_result, fp)

    datasetname = 'THUMOS14' if dataset == 'Thumos14reduced' else 'ActivityNet12'
    pwd = os.path.abspath(os.getcwd())
    gt_path = os.path.join(pwd, 'eval/gt.json') if dataset == 'Thumos14reduced' else os.path.join(pwd, 'eval/gt_anet12.json')
    aps = eval_mAP(gt_path, outpath, datasetname=datasetname, eval_class=eval_class_one, eval_vid=eval_vid)
    acc, top3, avg3 = eval_acc(gt_path, outpath, datasetname=datasetname, eval_class=picked_class)
    return aps, [acc, top3, avg3], t1, t3

if __name__ == '__main__':

    # config dataset
    dataset = 'Thumos14reduced'
    # dataset = 'ActivityNet1.2'
    config = Config(dataset)
    videodb = VideoDataset(dataset, config, split='cvpr18')

    test_idx = videodb.test_set
    train_class = videodb.train_cls
    test_class = videodb.test_cls

    # group_seg: group action segments by category
    # group_video: group videos by video category
    group_seg = videodb.groups_seg
    group_video = videodb.groups_video

    # 5-way (5-class) sample_num-shot (eg. 1-shot)
    sample_num=1

    ap_100 = []
    acc_100 = []
    hit1 = []
    hit3 = []
    for i in range(1000):
        random.seed(i*100)
        s = time.time()
        print(">>> Prepare sample set and test set")
        # Prepare sample set(Note: the reference videos in sample set are with arbitrary length)
        picked_class, train_idx, train_labels, sample_idx, sample_labels \
            = videodb.pick_class_ep(test_class, 5, 1, 0)
        test_pick_vids, test_pick_labels = videodb.pick_test_vid(picked_class, test_idx)
        sample_set = get_sample_set(videodb.feats, videodb.video_idxs, sample_idx, group_seg, picked_class, sample_num)
        # evaluate action recognition and localization performance using pretrained features
        aps, ac, top1, top3 = eval(videodb.feats, videodb.video_idxs, sample_set, dataset, picked_class, test_pick_vids, test_pick_labels,eval_one=True)

        ap_100.append(aps)
        acc_100.append(ac)
        hit1.append(top1)
        hit3.append(top3)
        e = time.time()
        print('Index: {}/{}; Time: {}s'.format(i, 1000, e-s))

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    ap_100 = np.array(ap_100)
    acc_100 = np.array(acc_100)
    ap = np.mean(ap_100, axis=0)

    print('map@0.5 of 1000 runnings is     {}'.format(ap[4]))
    print('mean, max, min of map@0.1:0.9   {} {} {}'.format(np.mean(ap),
                                                            np.max(np.mean(ap_100, axis=1)),
                                                            np.min(np.mean(ap_100, axis=1))))
    # print('hit@1 mean, max, min {} {} {}'.format(np.mean(acc_100[:, 0]),
    #                                              np.max(acc_100[:, 0]),
    #                                              np.min(acc_100[:, 0])))
    # print('hit@3 mean, max, min {} {} {}'.format(np.mean(acc_100[:, 1]),
    #                                                  np.max(acc_100[:, 1]),
    #                                                  np.min(acc_100[:, 1])))
    print('precision of hit 1 is {}'.format(np.mean(hit1)))
    print('precision of hit 3 is {}'.format(np.mean(hit3)))









