import numpy as np
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def group_video_by_cat(annos, classlist, subset='training'):
    '''Froup videos according their categories
    '''
    groups = dict()
    for cls in classlist:
        groups[cls] = []
    for vid in annos.keys():
        ss = annos[vid]['subset']
        if ss == subset:
            for anno in annos[vid]['annotations']:
                label = anno['label']
                groups[label].append([vid, anno['segment']])
    return groups

def result2json(result, cat):
    """Record the proposals to the json file
    """
    result_file = []

    for i in range(len(result)):
        line = {'label': cat, 'score': result[i,2],
                'segment': [result[i,0]*16/25.0, result[i,1]*16/25.0]}
        result_file.append(line)

    return result_file

def postprocess_anet(scores, slen):
    '''find the proposal with similar length with sample/reference video
    '''

    minimum = int(np.min(scores))
    maximum = max(int(np.max(scores)), minimum + 1)
    segment_final_predict = []
    best_len_diff = 1000

    for threshold in range(minimum, maximum):
        segment_predict = []
        avg_length = []
        tmp = scores
        vid_pred = np.concatenate([np.zeros(1), (tmp > threshold).astype('float32'), np.zeros(1)], axis=0)
        vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
        # start and end of proposals where segments are greater than the average threshold for the class
        s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
        e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]

        for j in range(len(s)):
            aggr_score = np.mean(tmp[s[j]:e[j]])
            # append proposal if length is at least 2 segments (16 frames segments @ 25 fps - around 1.25 second)
            if e[j] - s[j] >= 2:
                segment_predict.append([s[j], e[j], aggr_score])
                avg_length.append(e[j] - s[j] + 1)
        if np.abs(np.mean(avg_length) - slen) < best_len_diff:
            best_len_diff = np.abs(np.mean(avg_length) - slen)
            segment_final_predict = segment_predict
    segment_final_predict = np.array(segment_final_predict)
    return segment_final_predict

def postprocess(scores, activityNet, th=0):
    '''get proposals according to proposal score only
    '''
    segment_predict = []
    tmp = scores

    if not activityNet:
        threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*0.5
    else:
        threshold = th

    vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
    vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
    # start and end of proposals where segments are greater than the average threshold for the class
    s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
    e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
    for j in range(len(s)):
        # Original - Aggregate score is max value of prediction for the class in the proposal and 0.7 * mean(top-k) score of that class for the video
        aggr_score = np.mean(tmp[s[j]:e[j]])
        # append proposal if length is at least 2 segments (16 frames segments @ 25 fps - around 1.25 second)
        if e[j]-s[j]>=2:
           segment_predict.append([s[j], e[j], aggr_score])
    segment_predict = np.array(segment_predict)
    return segment_predict

def construct_res_dict(iteration, loss_cls):
    return {
        'iteration': iteration,
        'loss_cls': loss_cls
    }

def update_lr(optimizer, lr):
    """Set learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LossAccumulator:
    def __init__(self):
        self.loss_cls = np.array([], dtype=np.float32)
        self.loss_tot = np.array([], dtype=np.float32)

    def update(self, loss_cls):
        self.loss_cls = np.append(self.loss_cls, loss_cls)
        self.loss_tot = np.append(self.loss_tot, loss_cls)

    def get_loss_cls(self):
        return self.loss_cls

    def get_mean(self, n):
        if len(self.loss_cls) == 0:
            raise ValueError("Loss accumulator is empty.")
        elif len(self.loss_cls) >= 1:
            loss_cls_mean = self.loss_cls[-n:].mean()
            loss_tot_mean = self.loss_tot[-n:].mean()
            return loss_cls_mean, loss_tot_mean

    def get_std(self, n):
        if len(self.loss_cls) == 0:
            raise ValueError("Loss accumulator is empty.")
        elif len(self.loss_cls) >= 1:
            loss_cls_std = self.loss_cls[-n:].std()
            loss_tot_std = self.loss_tot[-n:].std()
            return loss_cls_std, loss_tot_std
