from __future__ import print_function

from scipy.spatial.distance import cosine
import numpy as np
from numpy import linalg as LA

# def cos(x,y):
#     cos = 0.5 *(1+ (np.inner(x,y))/(LA.norm(x)*LA.norm(y)))
#     return cos

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return  np.exp(x)/np.sum(np.exp(x))

def get_ssm_eu(feats, feats_p=[]):
    '''Compute temporal similarity matrix between feats and feats/feats_p with euclidean distance
    '''
    l = len(feats)
    if len(feats_p) == 0:
        feats_p = feats.copy()
    sm = np.zeros((len(feats), len(feats_p)))
    for i in range(len(feats)):
        for j in range(len(feats_p)):
            sm[i, j] = -np.sum((feats[i]-feats_p[j])**2)

    return sm

def get_ssm_ip(feats, feats_p=[]):
    '''Compute temporal similarity matrix between feats and feats/feats_p with dot product
    '''
    l = len(feats)
    if len(feats_p) == 0:
        feats_p = feats.copy()
    feats = feats[:,None]
    feats_p = feats_p[None,...]
    a = np.inner(feats, feats_p)
    a = np.squeeze(a)
    return a

def get_ssm_cos(feats, feats_p=[]):
    '''Compute temporal similarity matrix between feats and feats/feats_p with cosine similarity
    '''
    l = len(feats)
    feats = np.maximum(feats, 0)
    feats = feats/LA.norm(feats, axis=-1)[:,None]
    if len(feats_p) == 0:
        feats_p = feats.copy()
    else:
        feats_p = np.maximum(feats_p, 0)
        feats_p = feats_p/LA.norm(feats_p, axis=-1)[:,None]
    a = np.inner(feats, feats_p)
    return a
