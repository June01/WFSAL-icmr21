import numpy as np
import json
import os

class Config(object):
    """Most of the dataset configurations are saved here
    """
    def __init__(self, dataset):

        # path to current folder
        self.root = os.path.join(os.path.abspath(os.getcwd()), 'data')

        if dataset == 'ActivityNet1.2':
            # ActivityNet settings
            self.anno_dict_path = os.path.join(self.root,'activity_net.v1-2.min.json')
            video_idxs = np.load(os.path.join(self.root, 'ActivityNet1.2-Annotations/url.npy'))
            self.video_idxs = [vid.decode('utf-8').split('=')[-1] for vid in video_idxs]
        else:
            # Thumos14 settings
            self.anno_dict_path = os.path.join(self.root, 'gt_all_th14.json')
            video_idxs = np.load(os.path.join(self.root, 'Thumos14reduced-Annotations/videoname.npy'))
            self.video_idxs = [vid.decode('utf-8') for vid in video_idxs]


        # Initialize all the arguments
        print(">>> Loading features and annotations from {}".format(dataset))
        anno_dict = json.load(open(self.anno_dict_path))
        self.annos = anno_dict['database']

        self.feat_path =  os.path.join(self.root, '{}-I3D-JOINTFeatures.npy'.format(dataset))
        self.feats = np.load(self.feat_path, encoding='bytes', allow_pickle=True)
        classlist = np.load(os.path.join(self.root, '{}-Annotations/classlist.npy'.format(dataset)), allow_pickle=True)
        self.classlist = [clx.decode('utf-8') for clx in classlist]

        print(">>> Prepare test videos")
        subset = 'validation' if dataset == 'ActivityNet1.2' else 'testing'
        # Prepare test data
        self.test_set = []
        for vid in self.annos.keys():
            if self.annos[vid]['subset'] == subset:
                self.test_set.append(vid)

        self.lr_schedule = {
            1: 1e-4,
            1000: 5e-5,
            10000: 1e-5
        }