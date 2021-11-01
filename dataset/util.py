import numpy as np


def str2ind(categoryname, classlist):
   return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]

def strlist2indlist(strlist, classlist):
    return [str2ind(s,classlist) for s in strlist]

def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist,classlist)], axis=0)