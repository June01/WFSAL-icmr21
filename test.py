import torch
import json
import numpy as np
import os.path as osp
import time
import random
import torch.nn.functional as F

# user-defined libs
from dataset.dataset import get_data_loader
from util.utils import result2json, postprocess, postprocess_anet
from lib.loss import get_distmat
from eval.get_detection_performance import eval_mAP
from eval.get_classification_performance import eval_acc
from train import get_attention_mask, get_mean_feats

def test(args, videodb, fcencoder, attgen, models_dir, eval_class, activityNet=False):
    if args.verbose:
        print("#.Testing on {}...".format(args.dataset))

    ap_100 = []
    hit1 = []
    hit3 = []
    acc_100 = []
    with torch.no_grad():
        test_idx = videodb.test_set
        # test_idx.sort()
        test_labels = None

        for i in range(args.test_ep):
            # Prepare sample set
            picked_class, train_idx, train_labels, sample_idx, sample_labels \
                = videodb.pick_class_ep(eval_class, args.pick_class_num, args.sample_num_per_class, 0)
            sample_loader = get_data_loader(videodb, picked_class, args.sample_num_per_class, sample_idx, sample_labels,
                                            num_per_class=args.sample_num_per_class, split='sample', shuffle=False)

            samples, sample_labels, sidxs, sample_lens, sample_mask = sample_loader.__iter__().next()
            test_pick_vids, test_pick_labels = videodb.pick_test_vid(picked_class, test_idx)
            random.shuffle(test_pick_vids)
            test_loader = get_data_loader(videodb, picked_class, args.test_batch_size, test_pick_vids, test_labels,
                                          split='test', shuffle=True)
            if args.verbose:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("#.Testing episode {}/{}".format(i, args.test_ep))
                print("  \\__.Dataset")
                print("       \\__Name               : {}".format(args.dataset))
                print("       \\__Class Num          : {}".format(len(picked_class)))
                print("       \\__Picked_class       : {}".format(picked_class))
                print("       \\__Sample Num in all  : {}".format(len(sample_idx)))
                print("       \\__Batch  Num         : {}".format(args.test_batch_size))
                print("       \\__Test sample num    : {}".format(len(test_pick_vids)))
                print("       \\__samples.size()     : {}".format(samples.size()))
                print("       \\__sample.sum         : {}".format(torch.sum(samples)))
                print("       \\__samples idxs       : {}".format(sample_idx))

            if args.norm:
                print('Doing normalization of samples in testing')
                # batches = F.normalize(batches, dim=-1, p=2)
                samples = F.normalize(samples, dim=-1, p=2)

            if args.cuda and torch.cuda.is_available():
                samples = samples.cuda()
                sample_mask = sample_mask.cuda()

            if args.encoder:
                if args.cuda and torch.cuda.is_available():
                    samples = fcencoder(samples, is_training=False).cuda()
                else:
                    samples = fcencoder(samples, is_training=False)


            count = 0
            if args.verbose:
                print("   \\__Iterate test set")
            eval_vid = []
            eval_class_each = []
            annos_set = {}
            for batches, batch_labels, bidxs, batch_lens, batch_mask in test_loader:
                # if args.verbose:
                #     print("       \\__count             : {}/{}".format(count, len(test_pick_vids)))
                #     print("       \\__batches.size()    : {}".format(batches.size()))
                #     print("       \\__batches.sum()     : {}".format(torch.sum(batches)))
                #     print("       \\__test video        : {}".format(test_pick_vids[bidxs[0]]))
                #     print("       \\__test label        : {}".format(test_pick_labels[test_pick_vids[bidxs[0]]]))
                #     print("       \\__annotations       : {}".format(videodb.annos[test_pick_vids[bidxs[0]]]))
                #     print("===================================================")


                count += 1
                if torch.sum(batches) == 0 or test_pick_vids[bidxs[0]] == 'video_test_0000270' \
                        or  test_pick_vids[bidxs[0]] == 'video_test_0001496':
                    continue

                if args.cuda and torch.cuda.is_available():
                    batches = batches.cuda()
                    batch_lens = batch_lens.cuda()
                    batch_mask = batch_mask.cuda()

                if args.encoder:
                    if args.cuda and torch.cuda.is_available():
                        batches = fcencoder(batches, False).cuda()
                    else:
                        batches = fcencoder(batches, False)

                # print('test samples and batches {} {}'.format(samples.size(), batches.size()))
                mean_feats = get_mean_feats(samples, sample_lens, args.pick_class_num * args.sample_num_per_class)

                # atten_frame_level [1, class_num*sample_num_per_class, l]
                atten_frame_level = get_attention_mask(samples, batches, sample_mask, batch_mask, args,
                                                      args.pick_class_num, args.num_in, attgen, mode='test')

                # print('samples and batches size is {} {}'.format(samples.size(), batches.size()))
                # [n_batch, length, feature_dim]
                # print('Eu distance between temporal pooling batches and mean feature')
                distmat, _ = get_distmat(batches,
                                         atten_frame_level.transpose(1, 2),
                                         batch_lens.repeat(args.sample_num_per_class * args.pick_class_num),
                                         mean_feats,
                                         args.pick_class_num,
                                         distance=args.distance,
                                         device=torch.device("cuda"))

                bs, _, l = atten_frame_level.size()
                atten_frame_level = atten_frame_level.view(bs, -1, args.pick_class_num, l)
                atten_frame_level = torch.mean(atten_frame_level, axis=1)

                sm = torch.nn.Softmax(dim=1)
                distmat = sm(distmat)

                tmp, topkidx = torch.topk(distmat[0], k=args.pick_class_num, dim=0)

                annos = []
                topk_cls = []
                for j in range(3):
                    key = picked_class[topkidx[j]]
                    topk_cls.append(key)

                for j in range(1):
                    key = picked_class[topkidx[j]]
                    # topk_cls.append(key)
                    if not activityNet:
                        segments = postprocess(atten_frame_level[0, topkidx[j]].cpu().detach().numpy(), activityNet)

                    else:
                        segments = postprocess_anet(atten_frame_level[0, topkidx[j]].cpu().detach().numpy(),
                                              np.mean(sample_lens[topkidx[j]*args.sample_num_per_class:(topkidx[j]+1)*args.sample_num_per_class].cpu().detach().numpy()))
                    annos.extend(result2json(segments, key))

                if args.eval_one:
                    eval_vid.append(test_pick_vids[bidxs[0]])
                    for vlab in list(test_pick_labels[test_pick_vids[bidxs[0]]]):
                        if vlab in picked_class:
                            eval_class_each.append(vlab)
                    break

            outjson = {}
            outjson[test_pick_vids[bidxs[0]]] = annos
            final_result = dict()
            final_result['version'] = 'VERSION 1.2'
            final_result['external_data'] = []
            final_result['results'] = outjson

            outpath = 'output_{}_{}.json'.format(args.dataset, args.distance)
            with open(outpath, 'w') as fp:
                json.dump(final_result, fp)

            datasetname = 'THUMOS14' if args.dataset == 'Thumos14reduced' else 'ActivityNet12'
            gt_path = 'eval/gt.json' if args.dataset == 'Thumos14reduced' else 'eval/gt_anet12.json'

            # TODO: Add eval_vid to classification
            if args.eval_one:
                aps = eval_mAP(gt_path, outpath, datasetname=datasetname, eval_class=eval_class_each, eval_vid=eval_vid)
                if topk_cls[0] in eval_class_each:
                    hit1.append(1)
                    hit3.append(1)
                else:
                    hit1.append(0)
                    if (topk_cls[1] in eval_class_each) or (topk_cls[2] in eval_class_each):
                        hit3.append(1)
                    else:
                        hit3.append(0)
            else:
                aps = eval_mAP(gt_path, outpath, datasetname=datasetname, eval_class=picked_class, eval_vid=[])
                acc, top3, avg3 = eval_acc(gt_path, outpath, datasetname=datasetname, eval_class=picked_class)
                acc_100.append([acc, top3, avg3])

            ap_100.append(aps)

    ap_100 = np.array(ap_100)
    ap = np.mean(ap_100, axis=0)


    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if args.eval_one:
        acc_100 = np.zeros((1000, 3))
        np.save(osp.join(models_dir, 'acc_{}_{}_{}_{}.npy'.format(args.dataset, time_str, "%.4f"%np.mean(np.mean(hit1)), "%.4f"%np.mean(np.mean(hit3)))), acc_100)
        print('hit1 and hit3 of precision are {} {}'.format(np.mean(hit1), np.mean(hit3)))
    else:
        acc_100 = np.array(acc_100)
        np.save(osp.join(models_dir,
                         'acc_{}_{}_{}.npy'.format(args.dataset, time_str, "%.4f" % np.mean(np.mean(hit1)),
                                                      "%.4f" % np.mean(np.mean(hit3)))), acc_100)
    if activityNet:
        ap_5 = ap[0]
        print_result_anet(ap_100, acc_100, ap, models_dir, args.verbose)
    else:
        ap_5 = ap[4]
        print_result_th14(ap_100, acc_100, ap, models_dir, args.verbose)
    ap_mean = np.mean(ap)
    np.save(osp.join(models_dir, 'ap_{}_{}_{}_{}_{}.npy'.format(args.dataset, time_str, "%.4f" % ap_5, ap_mean, args.eval_one)), ap_100)

    return ap_100, acc_100

def print_result_th14(ap_100, acc_100, ap, models_dir, verbose=True):
    if verbose:
        print('Average map of {} iterations'.format(models_dir))
        print('mean of map@0.1 is         {}'.format(ap[0] * 100))
        print('mean of map@0.2 is         {}'.format(ap[1] * 100))
        print('mean of map@0.3 is         {}'.format(ap[2] * 100))
        print('mean of map@0.4 is         {}'.format(ap[3] * 100))
        print('mean of map@0.5 is         {}'.format(ap[4] * 100))
        print('mean of map@0.6 is         {}'.format(ap[5] * 100))
        print('mean of map@0.7 is         {}'.format(ap[6] * 100))
        print('mean of map@0.8 is         {}'.format(ap[7] * 100))
        print('mean of map@0.9 is         {}'.format(ap[8] * 100))

        print('mean, max, min of map@0.1:0.9   {} {} {}'.format(np.mean(ap) * 100,
                                                                np.max(np.mean(ap_100, axis=1)) * 100,
                                                                np.min(np.mean(ap_100, axis=1)) * 100))
        print('diff is {} {}'.format((np.max(np.mean(ap_100, axis=1)) - np.mean(np.mean(ap_100, axis=1))) * 100,
                                     (np.mean(np.mean(ap_100, axis=1)) - np.min(np.mean(ap_100, axis=1))) * 100))

def print_result_anet(ap_100, acc_100, ap, models_dir, verbose=True):
    if verbose:
        print('Average map of {} iterations'.format(models_dir))
        print('mean of map@0.1 is         {}'.format(ap[0] * 100))
        print('mean of map@0.2 is         {}'.format(ap[2] * 100))
        print('mean of map@0.3 is         {}'.format(ap[4] * 100))
        print('mean of map@0.4 is         {}'.format(ap[6] * 100))
        print('mean of map@0.5 is         {}'.format(ap[8] * 100))
        print('mean of map@0.55 is         {}'.format(ap[9] * 100))
        print('mean of map@0.6 is         {}'.format(ap[10] * 100))
        print('mean of map@0.65 is         {}'.format(ap[11] * 100))
        print('mean of map@0.7 is         {}'.format(ap[12] * 100))
        print('mean of map@0.75 is         {}'.format(ap[13] * 100))
        print('mean of map@0.8 is         {}'.format(ap[14] * 100))
        print('mean of map@0.85 is         {}'.format(ap[15] * 100))
        print('mean of map@0.9 is         {}'.format(ap[16] * 100))
        print('mean of map@0.95 is         {}'.format(ap[17] * 100))


        print('mean, max, min of map@0.5:0.95   {} {} {}'.format(np.mean(ap) * 100,
                                                                 np.max(np.mean(ap_100, axis=1)) * 100,
                                                                 np.min(np.mean(ap_100, axis=1)) * 100))
        print('diff is {} {}'.format((np.max(np.mean(ap_100, axis=1)) - np.mean(np.mean(ap_100, axis=1))) * 100,
                                     (np.mean(np.mean(ap_100, axis=1)) - np.min(np.mean(ap_100, axis=1))) * 100))

