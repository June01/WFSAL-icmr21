from __future__ import print_function
import sys
sys.path.append('..')

import argparse
import os.path as osp
import torch
from model import Model

import pickle as pkl
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import json
from numpy import linalg as LA

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import ssm

# from nms import nms

# def cos(x,y):
#     cos = 0.5 *(1+ (np.inner(x,y))/(LA.norm(x)*LA.norm(y)))
#     return cos

def plot_props(ax, pred_segs, gt_segs, dur=1, l=1, ranked_idx=[]):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    if len(ranked_idx) == 0:
        idx_list = range(len(pred_segs))
    else:
        idx_list = ranked_idx[:5]
    count = 0
    
    if not (dur == 1 and l == 1) and len(gt_segs)!=0:
        ax =  plot_gt(ax, gt_segs, [], dur, l)
    
    for idx in idx_list:
        item = pred_segs[idx]
        (s,e) = item['segment']
        new_s = s
        new_e = e
        length = new_e - new_s
        score = item['score']
        print("           \\__Start: {}({}) | End: {}({}) | Score: {}".format(new_s, s, new_e, e, score))
    #         Create a Rectangle patch
        rect = patches.Rectangle((new_s, new_s), length, length, linewidth=max(1, 5-count), edgecolor='cyan',facecolor='none')

#         Add the patch to the Axes
        ax.add_patch(rect)
        count += 1
    return ax

def plot(idx, dataset, mask = [], feats=[]):
    if len(feats) == 0:
        feats = dataset.features[idx]
    if len(mask) != 0:
        feats = mask[:,None]*feats
    l = feats.shape[0]
    # feats = cornerpooling(feats)
    # feats = normalize(feats, axis=0)
#     ssm_rgb = get_ssm_ip(cornerpooling(feats[:,:1024]))
#     ssm_flow = get_ssm_ip(cornerpooling(feats[:,1024:]))
    ssm_rgb = get_ssm_ip(feats[:,:1024])
    ssm_flow = get_ssm_ip(feats[:,1024:])
    ssm_all = get_ssm_ip(feats)

    # ssm = (ssm>mean)*1.0
    subset = dataset.subset[idx].decode('utf-8')
    dur = dataset.duration[idx]
    gt_segs = dataset.segments[idx]
    gt_labs = dataset.gtlabels[idx]
    if dataset.activity_net:
        url = dataset.urls[idx].decode('utf-8')
        vid = url.split('=')[-1]
    else:
        url = 'None'
        vid = dataset.videoname[idx]
    # print(url)
    

    # fig,ax = plt.subplots(1)
    # plt.imshow(ssm_rgb)

    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
    ax0 = axs[0]
    ax1 = axs[1]

    print("#.Sample index: {}".format(idx))
    print("  \\__Video ID                 : {}".format(vid))
    print("  \\__.Annotations:")
    print("      \\__Subset      : {}".format(subset))
    print("      \\__Duration    : {} s".format(dur))
    print("      \\__YouTube URL : {}".format(url))
    print("      \\__GT Segments :")

    for i in range(len(gt_segs)):
        item = gt_segs[i]
        (s,e) = item
        new_s = int(s/dur * l)
        new_e = int(e/dur * l)
        length = new_e - new_s
        label = gt_labs[i]
        print("           \\__Start: {}({}) | End: {}({}) | Class: {}".format(new_s, s, new_e, e, label))

    ax0 = plot_gt(ax0, gt_segs, gt_labs, dur, l)
    ax1 = plot_gt(ax1, gt_segs, gt_labs, dur, l)

    ax0.imshow(ssm_rgb)
    # ax.colorbar()
    ax0.set_title('ssm_rgb')
    ax1.imshow(ssm_flow)
    # ax.colorbar()
    ax1.set_title('ssm_flow')
#     print(gt_labs)
    return ssm_rgb, ssm_flow, ssm_all

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return  np.exp(x)/np.sum(np.exp(x))

def load_feature(vid, TEM_feat_path, gts_results):

    video_name = 'v_' +  vid + '.csv'
    video_feat_path = osp.join(TEM_feat_path, video_name)
    print(video_feat_path)
    if not osp.exists(video_feat_path) or gts_results[vid]['subset'] == 'testing':
        return [0]

    feats = np.array(pd.read_csv(video_feat_path))

    return feats

# Three methods to calculate the self-similarity matrix
# get_ssm_cosine: similarity is calculated by cosine distance
# get_ssm_ip(same as get_ssm_innerp): mean(inner product)
def get_ssm_cosine(feats, feats_p = []):
    '''Calculate the self-similarity matrix
    '''
    length = feats.shape[0]
    ssm = np.zeros((length, length))
    # cosine distance
    
    if len(feats_p) == 0:
        feats_p = feats
    for i in range(length):
        for j in range(length):
            # similarity_matrix[i,j] = np.sum(new_vector[i]*new_vector[j])
            ssm[i,j] = 1-cosine(feats[i], feats_p[j])
    return ssm

def get_ssm_exp(feats, feats_p = []):
    '''Calculate the self-similarity matrix
    '''
    length = feats.shape[0]
    ssm = np.zeros((length, length))
    # cosine distance
    
    if len(feats_p) == 0:
        feats_p = feats
    for i in range(length):
        for j in range(length):
            # similarity_matrix[i,j] = np.sum(new_vector[i]*new_vector[j])
#             ssm[i,j] = 1-cosine(feats[i], feats_p[j])
            ssm[i,j] = np.exp(- 0.001 * LA.norm(np.abs(feats[i]-feats_p[j]), 2))
    return ssm

def get_ssm_ip(feats, feats_p=[]):
    l = len(feats)
    if len(feats_p) == 0:
        feats_p = feats.copy()
    feats = feats[:,None]
    feats_p = feats_p[None,...]
#     print(feats.shape, feats_p.shape)
    a = np.inner(feats, feats_p)
    a = np.squeeze(a)
    return a

# def get_ssm_ip(feats, feats_p=[]):
#     l, d = feats.shape
#     ultimate2 = np.zeros((l,l))
#     if len(feats_p) == 0:
#         feats_p = feats
#     for i in range(l):
#         for j in range(l):
#             x = feats[i:i+1]
#             y = feats_p[j:j+1]
# #             print(x.shape, y.shape)
#             xy = np.dot(x, y.T)
#             ultimate1 = np.max(xy, axis=-1)
#             ultimate2[i, j] = np.mean(ultimate1, axis=-1)
#     return ultimate2

# def get_ssm_cos(feats, feats_p=[]):
#     l = len(feats)
#     feats = np.maximum(feats[:,None], 0)
#     feats = feats/LA.norm(feats, axis=-1)[:,None]
#     if len(feats_p) == 0:
#         feats_p = feats.copy()
#     else:
#         feats_p = np.maximum(feats_p[None,...], 0)
#         feats_p = feats_p/LA.norm(feats_p, axis=-1)[:,None]
# #     print(feats.shape, feats_p.shape)
# #     a = np.inner(feats, feats_p)
# #     a = np.squeeze(a)
#     ssm = np.zeros((l,l))
#     for i in range(l):
#         for j in range(l):
#             # similarity_matrix[i,j] = np.sum(new_vector[i]*new_vector[j])
#             ssm[i,j] = 1-cosine(feats[i], feats_p[j])

#     return ssm

def get_ssm_cos(feats, feats_p=[]):
    l = len(feats)
    feats = np.maximum(feats, 0)
    feats = feats/LA.norm(feats, axis=-1)[:,None]
    if len(feats_p) == 0:
        feats_p = feats.copy()
    else:
        feats_p = np.maximum(feats_p, 0)
        feats_p = feats_p/LA.norm(feats_p, axis=-1)[:,None]
#     print(feats.shape, feats_p.shape)
    a = np.inner(feats, feats_p)
    a = np.squeeze(a)
    return a

def cp(ssm):
    w, h = ssm.shape
    top = np.zeros((w,h))
    left = np.zeros((w,h))
    right = np.zeros((w,h))
    bottom = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            top[i,j] = np.max(ssm[i:,j])
            left[i,j] = np.max(ssm[i, j:])
            right[i,j] = np.max(ssm[i, :j+1])
            bottom[i,j] = np.max(ssm[:i+1, j])
    return (top+right+left+bottom)/4.0

def plot_gt(ax, gt_segs, gt_labs, dur, l):
    for i in range(len(gt_segs)):
        item = gt_segs[i]
        (s,e) = item
        new_s = int(s/dur * l)
        new_e = int(e/dur * l)
        length = new_e - new_s
        print("           \\__Start: {}({}) | End: {}({}) | l: {}".format(new_s, s, new_e, e, l))
    #         Create a Rectangle patch
        rect = patches.Rectangle((new_s, new_s), length, length, linewidth=1, edgecolor='red',facecolor='none')

#         Add the patch to the Axes
        ax.add_patch(rect)
    return ax

# def plot_props(ax, gt_segs):
#     for i in range(len(gt_segs)):
#         item = gt_segs[i]
#         (s,e) = item
#         new_s = s
#         new_e = e
#         length = new_e - new_s
#         label = gt_labs[i]
#         print("           \\__Start: {}({}) | End: {}({}) | Class: {}".format(new_s, s, new_e, e, label))
#     #         Create a Rectangle patch
#         rect = patches.Rectangle((new_s, new_s), length, length, linewidth=1, edgecolor='cyan',facecolor='none')

# #         Add the patch to the Axes
#         ax.add_patch(rect)
#     return ax

# atype is a flag of plot targets or predictions
def plot_ssm(idx, vid, dur, subset, url, ssm, gts=[], annos=[]):
    fig,ax = plt.subplots(1)
    plt.imshow(ssm)


    print("#.Sample index: {}".format(idx))
    print("  \\__Video ID                 : {}".format(vid))
    print("  \\__.Annotations:")
    print("      \\__Subset      : {}".format(subset))
    print("      \\__Duration    : {}".format(dur))
    print("      \\__YouTube URL : {}".format(url))
    print("      \\__GT Segments :")

    if gts != []:
        for item in gts:
            (s,e) = item['segment']
            new_s = int(s/dur * 100)
            new_e = int(e/dur * 100)
            l = new_e - new_s
            label = item['label']
            print("           \\__Start: {}({}) | End: {}({}) | Class: {}".format(new_s, s, new_e, e, label))
    #         Create a Rectangle patch
            rect = patches.Rectangle((new_s, new_s), l, l,linewidth=1, edgecolor='red',facecolor='none')

    #         Add the patch to the Axes
            ax.add_patch(rect)

    print("      \\__Pred Segments :")
    if annos != []:
        for item in annos:
            (s,e) = item['segment']
            new_s = int(s/dur * 100)
            new_e = int(e/dur * 100)
            
#             new_s = s
#             new_e = e
            
            l = new_e - new_s
            label = item['label']
            print("           \\__Start: {} | End: {} | Class: {} | Score: {}".format(new_s, new_e, label, item['score']))
    #         Create a Rectangle patch
            rect = patches.Rectangle((new_s, new_s), l, l,linewidth=1, edgecolor='cyan',facecolor='none')

    #         Add the patch to the Axes
            ax.add_patch(rect)

    plt.colorbar()
    plt.show()

# calculate gradient for ssm
def get_gssm(ssm):
    w = ssm.shape[0]
    gssm = np.zeros((w,w))
    for i in range(w):
        for j in range(w):
            c = ssm[i,j]
            x_left = max(0, i-1)
            x_right = min(w-1, i+1)
            y_left = max(0, j-1)
            y_right = min(w-1, j+1)

            d_left = abs(c-ssm[x_left, j])
            d_right = abs(c-ssm[x_right, j])
            d_above = abs(c-ssm[i, y_left])
            d_bottom = abs(c-ssm[i, y_right])

            gssm[i,j] = max(d_left, d_right, d_above, d_bottom)
    return gssm

def cornerpooling(feature):
    l, fs = feature.shape
    print(l,fs)
    tx = np.zeros((l, fs))
    ty = np.zeros((l, fs))
    for i in range(l):
        idx = l-i-1
        tx[idx] = np.max(feature[idx:], axis=0)
    for i in range(l):
        ty[i] = np.max(feature[:i+1], axis=0)
    return (tx+ty)/2.0

import math
def slidding_window_ssm(ssm, duration):
#     print(len(y1))
    props = []
    length = ssm.shape[0]
    for i in range(length):
        for j in range(i+5, length):
            l = j-i
#             expand_l = int(l/8)
            top = list(ssm[i:j, i])
            bottom = list(ssm[i:j, j])
            left = list(ssm[i,i:j])
            right = list(ssm[j, i:j])

            alls = top+bottom+left+right

            score = np.mean(alls)    
#             score = np.mean(ssm[i-expand_l:j+expand_l,i-expand_l:j+expand_l])-np.mean(ssm[i:j,i:j])
            props.append(
            {
                        'score': score,
                        'segment': [i, j],
                        'length': l,
                        'label': 'null'
            })

    return props

def find_video(classlist, c, labels):
    index_list = []
    c_name = classlist[c].decode('utf-8')
    for i in range(len(labels)):
        if c_name in labels[i]:
            index_list.append(i)
    return index_list

def rank(data,sort_key):
    # Make a list or ranks to be sorted
    ranks = [x for x in range(len(data))]
    # Sort ranks based on the key of data each refers to
    return sorted(ranks, reverse=True, key=lambda x:data[x][sort_key])

def plot_gtv2v(ax, gt_segs1, gt_segs2, gt_lab1, gt_lab2, lab, dur1, dur2, l1, l2):
    gt1 = []
    gt2 = []
    
    for i in range(len(gt_segs1)):
        label = gt_lab1[i]
        if label == lab:
            item = gt_segs1[i]
            (s,e) = item
            new_s = int(s/dur1 * l1)
            new_e = int(e/dur1 * l1)
            gt1.append([new_s, new_e])
    for i in range(len(gt_segs2)):
        label = gt_lab2[i]
        if label == lab:
            item = gt_segs2[i]
            (s,e) = item
            new_s = int(s/dur2 * l2)
            new_e = int(e/dur2 * l2)
            gt2.append([new_s, new_e])
    for [x1, x2] in gt2:
        for [y1, y2] in gt1:
            rect = patches.Rectangle((x1, y1), x2-x1+1, y2-y1+1, linewidth=1, edgecolor='red',facecolor='none')
            print(x1,y1, x2, y2)
#         Add the patch to the Axes
            ax.add_patch(rect)
    return ax

def plot_v2v_ssm(dataset, id1, id2, lab, cosine=False):
    
    feats1 = dataset.features[id1]
    feats2 = dataset.features[id2]
  
    l1 = feats1.shape[0]
    l2 = feats2.shape[0]

    if not cosine:
        ssm1 = get_ssm_ip(feats1)
        ssm2 = get_ssm_ip(feats2)
        ssm_v2v = get_ssm_ip(feats1, feats2)
    else:
        ssm1 = get_ssm_cos(feats1)
        ssm2 = get_ssm_cos(feats2)
        ssm_v2v = get_ssm_cos(feats1, feats2)

    gt_segs1 = dataset.segments[id1]
    gt_segs2 = dataset.segments[id2]
    
    gt_labs1 = dataset.gtlabels[id1]
    gt_labs2 = dataset.gtlabels[id2]
    
    dur1 = dataset.duration[id1]
    dur2 = dataset.duration[id2]

    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,10))
    ax0 = axs[0];
    ax1 = axs[1];
    ax2 = axs[2];

    print("#Similarity Matrix between {} and {}".format(id1, id2))
    
    ax0 = plot_gt(ax0, gt_segs1, gt_labs1, dur1, l1)
    ax1 = plot_gt(ax1, gt_segs2, gt_labs2, dur2, l2)
    ax2 = plot_gtv2v(ax2, gt_segs1, gt_segs2, gt_labs1, gt_labs2, lab, dur1, dur2,l1, l2)
#     print(gt_labs1, gt_labs2)

    ax0.imshow(ssm1)
    ax0.set_title('ssm of {}'.format(id1))
    
    ax1.imshow(ssm2)
    ax1.set_title('ssm of {}'.format(id2))

    ax2.imshow(ssm_v2v)
    ax2.set_title('ssm')
    ax2.set_xlabel('video {}'.format(id2))
    ax2.set_ylabel('video {}'.format(id1))
#     plt.show()
    
    return ssm1, ssm2, ssm_v2v
