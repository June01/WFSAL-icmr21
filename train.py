import os.path as osp
import torch

from dataset.dataset import get_data_loader
from lib.loss import get_loss
from util.utils import update_lr

import torch.nn.functional as F

def get_tsm_cos(samples_enc, batches_enc, args, pick_class_num, mode='train', norm=False):
    # each batch sample link to every samples to calculate similarities
    if mode == 'train':
        samples = samples_enc.unsqueeze(0).repeat(args.batch_num_per_class * pick_class_num, 1, 1, 1)
        batches = batches_enc.unsqueeze(0).repeat(args.sample_num_per_class * pick_class_num, 1, 1, 1)
    else:
        samples = samples_enc.unsqueeze(0).repeat(args.test_batch_size, 1, 1, 1)
        batches = batches_enc.unsqueeze(0).repeat(args.sample_num_per_class * pick_class_num, 1, 1, 1)

    batches = torch.transpose(batches, 0, 1)
    # print('=====feature ext=====')
    # samples: [sample_num_per_class * pick_class_num, pick_class_num, n, num_dim]
    # batches: [sample_num_per_class * pick_class_num, pick_class_num, m, num_dim]
    # print(samples.size(), batches.size())

    if norm:
        batches = batches / (batches.norm(dim=-1)[..., None].repeat(1, 1, 1, samples_enc.size(-1)) + 1e-5)
        samples = samples / (samples.norm(dim=-1)[..., None].repeat(1, 1, 1, samples_enc.size(-1)) + 1e-5)
    # if mode=='test':
    # print(batches.size(), samples.size())
    # import pdb
    # pdb.set_trace()
    tsm = torch.matmul(batches, samples.transpose(-2, -1))

    return tsm

def get_tsm_eu(samples_enc, batches_enc, args, pick_class_num, mode='train', device=torch.device("cuda")):

    len_bat = batches_enc.size(1)
    len_sam = samples_enc.size(1)
    d = batches_enc.size(-1)
    num_b = len(batches_enc)
    num_s = len(samples_enc)
    # print(samples_enc.shape, batches_enc.shape, len_bat, len_sam)

    # each batch sample link to every samples to calculate similarities
    if mode == 'train':
        samples = samples_enc.unsqueeze(0).repeat(args.batch_num_per_class * pick_class_num, 1, 1, 1)
        batches = batches_enc.unsqueeze(1).repeat(1, args.sample_num_per_class * pick_class_num, 1, 1)
    else:
        samples = samples_enc.unsqueeze(0).repeat(args.test_batch_size, 1, 1, 1)
        batches = batches_enc.unsqueeze(1).repeat(1, args.sample_num_per_class * pick_class_num, 1, 1)

    batches = batches.view(-1, len_bat, d)
    samples = samples.view(-1, len_sam, d)

    tsm_final = torch.cdist(batches, samples, p=2)
    tsm_final = tsm_final.view(num_b, num_s, len_bat, len_sam)

    return -tsm_final

def get_mean_feats(samples, sample_lens, num, device=torch.device("cuda")):
    mean_sample = torch.zeros(0).to(device)
    for i in range(len(samples)):
        sam_feat = samples[i, :sample_lens[i]]
        mean_sample = torch.cat([mean_sample, torch.mean(sam_feat, dim=0, keepdim=True)], dim=0)
    return mean_sample

def get_attention_mask(samples, batches, sample_mask, batch_mask, args, num_class, num_in, comparator=[], mode='train'):

    mask = get_tsm_cos(sample_mask, batch_mask, args, num_class, mode, norm=False)
    if num_in == 1:
        attention_mask = get_attention_mask_1(mask, samples, batches, args, num_class, mode)
    elif num_in == 2:
        attention_mask = get_attention_mask_2(mask, samples, batches, args, num_class, comparator, mode)
    elif num_in == 4:
        attention_mask = get_attention_mask_4(mask, samples, batches, args, num_class, comparator, mode)
    elif num_in == 6:
        attention_mask = get_attention_mask_6(mask, samples, batches, args, num_class, comparator, mode)

    return attention_mask

def get_attention_mask_6(mask, samples, batches, args, num_class, comparator=[], mode='train'):
    samples_flow = samples[:, :, :args.num_dim]
    samples_rgb = samples[:, :, args.num_dim:]

    batches_flow = batches[:, :, :args.num_dim]
    batches_rgb = batches[:, :, args.num_dim:]

    # mask = get_tsm_cos(sample_mask, batch_mask, args, num_class, mode, norm=False)

    tsm_rgb = get_tsm_cos(samples_rgb, batches_rgb, args, num_class, mode, norm=True)
    tsm_masked_rgb = tsm_rgb * mask

    tsm_flow = get_tsm_cos(samples_flow, batches_flow, args, num_class, mode, norm=True)
    tsm_masked_flow = tsm_flow * mask

    # [bs, num_class, L]
    atten_tsm_rgb, _ = torch.max(tsm_masked_rgb, dim=-1, keepdim=True)
    atten_tsm_flow, _ = torch.max(tsm_masked_flow, dim=-1, keepdim=True)

    ssm_rgb = get_tsm_cos(samples_rgb, batches_rgb, args, num_class, mode, norm=False)
    ssm_masked_rgb = ssm_rgb * mask
    atten_ssm_rgb, _ = torch.max(ssm_masked_rgb, dim=-1, keepdim=True)

    ssm_flow = get_tsm_cos(samples_flow, batches_flow, args, num_class, mode, norm=False)
    ssm_masked_flow= ssm_flow * mask
    atten_ssm_flow, _ = torch.max(ssm_masked_flow, dim=-1, keepdim=True)

    eu_rgb = get_tsm_eu(samples_rgb, batches_rgb, args, num_class, mode)
    eu_masked_rgb = eu_rgb * mask
    atten_eu_rgb, _ = torch.max(eu_masked_rgb, dim=-1, keepdim=True)

    eu_flow = get_tsm_eu(samples_flow, batches_flow, args, num_class, mode)
    eu_masked_flow = eu_flow * mask
    atten_eu_flow, _ = torch.max(eu_masked_flow, dim=-1, keepdim=True)

    # print(atten_tsm.size(), atten_ssm.size())
    atten_frame_level = comparator(torch.cat([atten_ssm_rgb, atten_ssm_flow, atten_tsm_rgb, atten_tsm_flow, atten_eu_rgb, atten_eu_flow], dim=-1))
    return atten_frame_level

def get_attention_mask_4(mask, samples, batches, args, num_class, comparator=[], mode='train'):

    samples_flow = samples[:, :, :args.num_dim]
    samples_rgb = samples[:, :, args.num_dim:]

    batches_flow = batches[:, :, :args.num_dim]
    batches_rgb = batches[:, :, args.num_dim:]

    # mask = get_tsm_cos(sample_mask, batch_mask, args, num_class, mode, norm=False)

    tsm_rgb = get_tsm_cos(samples_rgb, batches_rgb, args, num_class, mode, norm=True)
    tsm_masked_rgb = tsm_rgb * mask

    tsm_flow = get_tsm_cos(samples_flow, batches_flow, args, num_class, mode, norm=True)
    tsm_masked_flow = tsm_flow * mask

    # [bs, num_class, L]
    atten_tsm_rgb, _ = torch.max(tsm_masked_rgb, dim=-1, keepdim=True)
    atten_tsm_flow, _ = torch.max(tsm_masked_flow, dim=-1, keepdim=True)

    ssm_rgb = get_tsm_cos(samples_rgb, batches_rgb, args, num_class, mode, norm=False)
    ssm_masked_rgb = ssm_rgb * mask
    atten_ssm_rgb, _ = torch.max(ssm_masked_rgb, dim=-1, keepdim=True)

    ssm_flow = get_tsm_cos(samples_flow, batches_flow, args, num_class, mode, norm=False)
    ssm_masked_flow= ssm_flow * mask
    atten_ssm_flow, _ = torch.max(ssm_masked_flow, dim=-1, keepdim=True)

    # print(atten_tsm.size(), atten_ssm.size())
    atten_frame_level = comparator(torch.cat([atten_ssm_rgb, atten_ssm_flow, atten_tsm_rgb, atten_tsm_flow], dim=-1))


    return atten_frame_level

def get_attention_mask_2(mask, samples, batches, args, num_class, comparator=[], mode='train'):
    samples_flow = samples[:, :, :args.num_dim]
    samples_rgb = samples[:, :, args.num_dim:]

    batches_flow = batches[:, :, :args.num_dim]
    batches_rgb = batches[:, :, args.num_dim:]

    # [40, 5, 926, 22]
    # mask = get_tsm_cos(sample_mask, batch_mask, args, num_class, mode, norm=False)

    if args.tsm == 'cosine':
        tsm_rgb = get_tsm_cos(samples_rgb, batches_rgb, args, num_class, mode, norm=True)
        tsm_flow = get_tsm_cos(samples_flow, batches_flow, args, num_class, mode, norm=True)
    elif args.tsm == 'ip':
        tsm_rgb = get_tsm_cos(samples_rgb, batches_rgb, args, num_class, mode, norm=False)
        tsm_flow = get_tsm_cos(samples_flow, batches_flow, args, num_class, mode, norm=False)
    elif args.tsm == 'eu':
        tsm_rgb = get_tsm_eu(samples_rgb, batches_rgb, args, num_class, mode)
        tsm_flow = get_tsm_eu(samples_flow, batches_flow, args, num_class, mode)

    rgb_masked = tsm_rgb * mask
    flow_masked = tsm_flow * mask

    # [bs, num_class, L]
    atten_tsm_rgb, _ = torch.max(rgb_masked, dim=-1, keepdim=True)
    atten_tsm_flow, _ = torch.max(flow_masked, dim=-1, keepdim=True)

    # print(atten_tsm.size(), atten_ssm.size())
    atten_frame_level = comparator(torch.cat([atten_tsm_rgb, atten_tsm_flow], dim=-1))

    return atten_frame_level

def get_attention_mask_1(mask, samples, batches, args, num_class, mode):
    # mask = get_tsm_cos(sample_mask, batch_mask, args, num_class, mode, norm=False)
    if args.tsm == 'cosine':
        tsm = get_tsm_cos(samples, batches, args, num_class, mode, norm=True)
    elif args.tsm=='ip':
        tsm = get_tsm_cos(samples, batches, args, num_class, mode, norm=False)
    elif args.tsm == 'eu':
        tsm = get_tsm_eu(samples,batches, args, num_class, mode)

    tsm_masked = tsm * mask
    atten_frame_level, _ = torch.max(tsm_masked, dim=-1)

    return atten_frame_level

def train(fcencoder, lr_schedule, optimizer, videodb, epis, args, loss_accumulator, attgen, optimizer_filter, train_class):
    if epis in lr_schedule:
        if args.encoder:
            update_lr(optimizer, lr_schedule[epis])
        if args.num_in > 1:
            update_lr(optimizer_filter, lr_schedule[epis])

    # prepare data for each episode
    picked_class, train_idx, train_labels, sample_idx, sample_labels \
        = videodb.pick_class_ep(train_class, args.pick_class_num, args.sample_num_per_class, args.batch_num_per_class)

    sample_loader = get_data_loader(videodb, picked_class, args.sample_num_per_class, sample_idx, sample_labels,
                                    num_per_class=args.sample_num_per_class, split='sample', shuffle=False)
    batch_loader = get_data_loader(videodb, picked_class, args.batch_num_per_class, train_idx, train_labels,
                                   num_per_class=args.batch_num_per_class, split='train', shuffle=True)

    samples, sample_labels, sidxs, sample_lens, sample_mask = sample_loader.__iter__().next()
    batches, batch_labels, bidxs, batch_lens, batch_mask = batch_loader.__iter__().next()

    flag = False
    for j in range(len(samples)):
        if torch.sum(samples[j]) == 0:
            flag = True
            break
    for j in range(len(batches)):
        if torch.sum(batches[j]) == 0:
            flag = True
            break
    if flag:
        return

    if args.cuda and torch.cuda.is_available():
        samples = samples.cuda(); sample_mask = sample_mask.cuda()
        batches = batches.cuda(); batch_labels = batch_labels.cuda(); batch_lens = batch_lens.cuda();
        batch_mask = batch_mask.cuda()

    if args.encoder:
        optimizer.zero_grad()
    if args.num_in > 1:
        optimizer_filter.zero_grad()

    # use encoder to get feature representation
    if args.encoder:
        if args.cuda and torch.cuda.is_available():
            samples = fcencoder(samples).cuda()
            batches = fcencoder(batches).cuda()
        else:
            samples = fcencoder(samples)
            batches = fcencoder(batches)

    # get mean feature of reference/sample videos
    mean_feats = get_mean_feats(samples, sample_lens, args.pick_class_num * args.sample_num_per_class)

    # Given several query videos, infer attention mask using reference/sample videos.
    atten_frame_level = get_attention_mask(samples, batches, sample_mask, batch_mask, args,
                                                     args.pick_class_num, args.num_in, attgen)

    # print('Eu distance between temporal pooling batches and mean feature')
    loss_cls = get_loss(batches,
                       atten_frame_level.transpose(1, 2),
                       torch.eye(args.pick_class_num)[batch_labels],
                       batch_lens.repeat(args.sample_num_per_class * args.pick_class_num),
                       mean_feats,
                       args.pick_class_num,
                       distance=args.distance)

    loss = loss_cls

    loss_accumulator.update(loss_cls.data.item())
    print('Episodes: {}/{} loss_cls: {}, loss: {}'.format(epis, args.num_episodes, "%.6f" % loss_cls, "%.6f" %loss))
    loss.backward()
    if args.encoder:
        optimizer.step()
    if args.num_in > 1:
        optimizer_filter.step()