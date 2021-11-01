import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# user-defined libs
from dataset.dataset import VideoDataset
from lib.networks import AttentionGenerator, FCEncoder
# The config folder should be in consistent with each other
from configs.config import Config
from util.utils import LossAccumulator
from train import train
from test import test
from util.utils import construct_res_dict
# from adabelief_pytorch import AdaBelief
import time
import json

def parse_args():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Weakly few-shot learning training script")
    parser.add_argument('-v', '--verbose', action='store_true', default=True, help="increase output verbosity")
    # =============================== Data =============================================================================
    parser.add_argument('--dataset', type=str, default='Thumos14reduced', choices=['Thumos14reduced', 'ActivityNet1.2'], help="select training dataset")
    parser.add_argument('--num_workers', type=int, default=4, help="number of CPU workers used in data loading")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffle data")
    parser.set_defaults(shuffle=True)
    # =============================== Training =========================================================================
    # c-way k-shot n-query [[5, 1, 19], [5, 5, 15], [20, 1, 10], [20, 5, 5]]
    parser.add_argument("--pick_class_num", type=int, default=5)
    parser.add_argument("--sample_num_per_class", type=int, default=1)
    parser.add_argument("--batch_num_per_class", type=int, default=8)
    parser.add_argument("--num_dim", type=int, default=128)
    parser.add_argument("--num_episodes", type=int, default=5001)

    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam', 'adabelief'])
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum value for optimisation")
    parser.add_argument('--ckp_freq', default=2500, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--paral', dest='paral', action='store_true',
                        help="use data parallelism (multiple GPUs) during training")
    parser.set_defaults(paral=True)
    parser.add_argument('--no-paral', dest='paral', action='store_false',
                        help="do NOT use data parallelism (multiple GPUs) during training -- use first GPU available")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    parser.add_argument('--distance', type=str, choices=['cosine', 'l2', 'eu'], default='eu')
    parser.add_argument('--tsm', type=str, choices=['cosine', 'eu', 'ip'], default='cosine', help="The way to make  comparison in calculating segment-to-segment similarity")
    parser.add_argument('--lambda_cent', default=0, type=float, help='lambda for center loss')
    parser.add_argument('--lambda_ent', default=0, type=float, help='lambda for entropy loss')
    parser.add_argument('--lambda_contr', default=0, type=float, help='lambda for contrastive loss')
    parser.add_argument('--norm', dest='norm', action='store_true')
    parser.add_argument('--no-norm', dest='norm', action='store_false')
    parser.set_defaults(norm=False)
    parser.add_argument('--encoder', dest='encoder', action='store_true', help='Using video encoder')
    parser.add_argument('--no-encoder', dest='encoder', action='store_false', help='Using video encoder')
    parser.set_defaults(encoder=True)
    parser.add_argument('--bn', default=True, action='store_true', help='Decide if batch normalization is needed')
    parser.add_argument('--split', default='cvpr18', choices=[None, 'cvpr18', 'reverse'], help='This is used to split training and testing class')
    parser.add_argument('--num_in', choices=[1, 2, 4, 6], type=int, help='Number of attention generator input')
    # ================================ Testing =========================================================================
    parser.add_argument('--mode', type=str, dest='mode', default='training', choices=['training', 'testing'])
    parser.add_argument('--test_ep', default=1000, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--load', type=str, help="load pre-trained model")
    parser.add_argument('--models_root', default="models/", help="root directory for saving trained models")
    parser.add_argument('--postfix', default='test', help="postfix to discriminate different iterations")
    parser.add_argument('--eval_one', default=True, action='store_true', help='If evaluate on one video or many')
    args = parser.parse_args()

    return args

def save_result(checkpoint_model_filename, epis, loss_accumulator, fcencoder, args, cfg, attgen=[], acc=[], map=[]):

    checkpoint_results_dict = construct_res_dict(iteration=epis,
                                                 loss_cls=loss_accumulator.get_loss_cls())

    attgen_state_dict = attgen.module.state_dict() if args.paral else attgen.state_dict() if args.num_in > 1 else []
    fcencoder_state_dict = fcencoder.module.state_dict() if args.paral else fcencoder.state_dict() if args.encoder else []

    torch.save({'state_dict': fcencoder_state_dict,
                'state_dict_filter': attgen_state_dict,
                'args': args,
                'cfg': cfg,
                'acc': np.array(acc),
                'map': np.array(map),
                'results': checkpoint_results_dict}, checkpoint_model_filename)


def main():
    """training/testing script.
    Options:
        --dataset       : set training dataset ('ActivityNet1.2', 'thumos14')
        --models_root   : set root directory of models directories
        --batch_size    : set training batch size
        --num_workers   : set number of CPU workers during data loading
        --cuda          : use CUDA during training
        --no-cuda       : do NOT use CUDA during training
    """

    ################################################################################
    ##                                [ Load Data ]                               ##
    ################################################################################
    args = parse_args()
    activityNet = False if args.dataset == 'Thumos14reduced' else True

    cfg = Config(args.dataset)
    lr_schedule = cfg.lr_schedule

    if args.verbose:
        print("#.Load dataset: {}...".format(args.dataset))

    # Whole dataset, needs to sample in each episode
    videodb = VideoDataset(args.dataset, cfg, split=args.split)
    num_class = len(videodb.classlist)
    train_class = videodb.train_cls
    num_class_train = len(train_class)
    test_class = videodb.test_cls
    num_class_test = len(test_class)

    # Create output models directory under args.models_root
    # Models dir format: <models_root>/
    model_basename = "{}_{}way_{}shot_split_{}_enc_{}_tsm_{}_lcent_{}_indim_{}_{}"\
        .format(args.dataset,
                args.pick_class_num,
                args.sample_num_per_class,
                args.split,
                args.encoder,
                args.tsm,
                args.lambda_cent,
                args.num_in,
                args.postfix)

    models_dir = osp.join(args.models_root, model_basename)
    if not osp.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    ################################################################################
    ##                             [ Build Network ]                              ##
    ################################################################################
    # Set default tensor type
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Build network
    if args.encoder:
        fcencoder = FCEncoder(n_feature=2048, out_dim=args.num_dim)
        # Print parameter numbers
        if args.verbose:
            model_params = filter(lambda p: p.requires_grad, fcencoder.parameters())
            num_params = sum([np.prod(p.size()) for p in model_params])
            print("  \\__.Number of parameters encoder: {}m".format(num_params / pow(10, 6)))
    else:
        fcencoder = []
        args.num_dim = 1024
    if args.num_in>1:
        attgen = AttentionGenerator(args.num_in, args.bn)
        if args.verbose:
            model_params_filter = filter(lambda p: p.requires_grad, attgen.parameters())
            num_params_filter = sum([np.prod(p.size()) for p in model_params_filter])
            print("  \\__.Number of parameters filter : {}".format(num_params_filter))
    else:
        attgen = []

    # Initialize network with pre-trained model
    if args.load:
        if args.verbose:
            print("#. Load network with pre-trained model: {}".format(args.load))
        model = torch.load(args.load, map_location=lambda storage, loc: storage)

        model_args = model['args']
        model_cfg = model['cfg']
        if args.encoder:
            model_state_dict = model['state_dict']
            fcencoder.load_state_dict(model_state_dict)
            fcencoder.eval()
        # import pdb
        # pdb.set_trace()
        if args.num_in>1:
            model_state_dict_filter = model['state_dict_filter']
            attgen.load_state_dict(model_state_dict_filter)
            attgen.eval()

        if args.mode=='testing':
            args = model_args
            args.mode = 'testing'
        else:
            args = model_args
        cfg = model_cfg

    # Print options
    if args.verbose:
        print("#.xxx")
        print("  \\__.Dataset")
        print("       \\__Name                            : {}".format(args.dataset))
        print("       \\__CPU workers                     : {}".format(args.num_workers))
        print("       \\__Class Num of all, train, test   : {}/{}/{}".format(num_class,
                                                                             num_class_train,
                                                                             num_class_test))
        print("      \\__training class is                : {}".format(train_class))
        print("      \\__testing class is                 : {}".format(test_class))
        print("  \\__.Training parameters for each training episode")
        print("       \\__Pick class Num                  : {}".format(args.pick_class_num))
        print("       \\__Sample Num                      : {}".format(args.sample_num_per_class))
        print("       \\__Batch  Num                      : {}".format(args.batch_num_per_class))
        print("       \\__Number of episodes              : {}".format(args.num_episodes))
        print("       \\__LR schedule                     : {}".format(lr_schedule))
        print("       \\__Weight decay                    : {}".format(args.weight_decay))
        print("       \\__Use CUDA                        : {}".format(args.cuda))
        print("  \\__.Output")
        print("       \\__Models directory                : {}{}".format(args.models_root, model_basename))
        print("  \\__.Options")
        print("       \\__Use Encoder                     : {}".format(args.encoder))
        print("       \\__Use entropy loss                : {}".format(args.lambda_ent))
        print("       \\__Use Attention generator         : {}".format(args.num_in))
        print("       \\__TSM                             : {}".format(args.tsm))
        print("       \\__Distance metric                 : {}".format(args.distance))

    # Parallelize data over multiple GPUs in the batch dimension (if available) and enable benchmark mode in cudnn.
    if args.encoder:
        if args.cuda and torch.cuda.is_available():
            cudnn.benchmark = True
            if args.paral:
                fcencoder = torch.nn.DataParallel(fcencoder).cuda()
            else:
                fcencoder = fcencoder.cuda()


    if args.num_in > 1:
        if args.cuda and torch.cuda.is_available():
            cudnn.benchmark = True
            attgen = torch.nn.DataParallel(attgen).cuda()
        else:
            attgen = attgen.cuda()

    if args.mode == 'training':
        if args.encoder:
            optimizer = torch.optim.Adam(fcencoder.parameters(), lr=lr_schedule[1], weight_decay=args.weight_decay)
        else:
            optimizer = []
        if args.num_in > 1:
            optimizer_filter = torch.optim.Adam(attgen.parameters(), lr=lr_schedule[1], weight_decay=args.weight_decay)
        else:
            optimizer_filter = []

        loss_accumulator = LossAccumulator()
        best_acc = 0

        if args.verbose:
            print("#.Training on {}...".format(args.dataset))

        for epis in range(args.num_episodes):
            if args.encoder:
                fcencoder.train()
            if args.num_in > 1:
                attgen.train()
            train(fcencoder,
                  lr_schedule,
                  optimizer,
                  videodb,
                  epis, args,
                  loss_accumulator,
                  attgen,
                  optimizer_filter,
                  train_class)
            if epis % args.ckp_freq == 0 and epis > 0:
                checkpoint_model_filename = osp.join(models_dir, "cp.pth")
                save_result(checkpoint_model_filename, epis, loss_accumulator, fcencoder, args, cfg, attgen)
                if args.encoder:
                    fcencoder.eval()
                if args.num_in > 1:
                    attgen.eval()
                ap_100, acc_100 = test(args,
                                       videodb,
                                       fcencoder,
                                       attgen,
                                       models_dir,
                                       test_class,
                                       activityNet)
                if best_acc < np.mean(ap_100):
                    best_acc = np.mean(ap_100)
                    checkpoint_model_filename = osp.join(models_dir, "best.pth")
                    save_result(checkpoint_model_filename,
                                epis,
                                loss_accumulator,
                                fcencoder,
                                args,
                                cfg,
                                attgen,
                                acc_100,
                                ap_100)

    else:
        print('I am doing testing')
        ap_100, acc_100 = test(args,
                               videodb,
                               fcencoder,
                               attgen,
                               models_dir,
                               test_class,
                               activityNet)

if __name__ == '__main__':
    main()
