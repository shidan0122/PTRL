from __future__ import print_function
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from vocab import deserialize_vocab
import os
import train
from model import SGRAF
import data_f30k
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name="", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)

def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w2).clamp(min=eps)).squeeze()

def inter_relations(attn, batch_size, sourceL, queryL, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """
    attn = nn.LeakyReLU(0.1)(attn) 
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL) 
    attn = nn.Softmax(dim=1)(attn * xlambda) 
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn    

def intra_relation(K, Q, xlambda):
    """
    Q: (n_context, sourceL, d)
    K: (n_context, sourceL, d)
    return (n_context, sourceL, sourceL)
    """
    batch_size, KL = K.size(0), K.size(1)
    K = torch.transpose(K, 1, 2).contiguous()
    attn = torch.bmm(Q, K)

    attn = attn.view(batch_size*KL, KL)
    attn = nn.Softmax(dim=1)(attn*xlambda)
    attn = attn.view(batch_size, KL, KL)
    return attn


def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)

def TripletLoss(im, s, margin):
    scores = im @ s.T
    #scores = np.dot(im, s.T)
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    # keep the maximum violating negative for each query

    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()

def shard_attn_scores(net, img_embs, cap_embs, cap_lens, opt, shard_size=1000):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1
    net.eval()
    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim, _ = net.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    return sims
    
def xattn_score_test(images, captions, cap_lens, args):
    """
    Note that there is no need to perform sampling and updating operations in the test stage, 
    so we only keep the part except the Discriminative Mismatch Mining module as this test function.

    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    batch_size = n_image

    args.using_intra_info = True

    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        # --> (batch, d, sourceL)
        contextT = torch.transpose(images, 1, 2)

        # attention matrix between all text words and image regions
        attn = torch.bmm(cap_i_expand, contextT)

        attn_i = torch.transpose(attn, 1, 2).contiguous()
        attn_thres = attn - torch.ones_like(attn) * args.thres_safe
        # attn_thres = attn - torch.ones_like(attn) * opt.thres

        # Neg-Pos Branch Matching
        # negative attention 
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)
        if  args.using_intra_info:
            attn_intra = intra_relation(cap_i, cap_i, 5)
            attn_intra = attn_intra.repeat(batch_size, 1, 1)
            Row_max_intra = torch.bmm(attn_intra, Row_max.reshape(batch_size, n_word).unsqueeze(-1)).reshape(batch_size * n_word, 1)
            attn_neg = Row_max_intra.lt(0).double()
            t2i_sim_neg = Row_max * attn_neg
        else:
            attn_neg = Row_max.lt(0).float()
            t2i_sim_neg = Row_max * attn_neg

        # negative effects
        t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)


        # positive attention 
        # 1) positive effects based on aggregated features
        attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, args.lambda_softmax)
        weiContext_pos = torch.bmm(attn_pos, images)
        t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)

        # 2) positive effects based on relevance scores
        attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, args.lambda_softmax)
        t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)

        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r


        t2i_sim =  t2i_sim_neg + t2i_sim_pos
        sim = t2i_sim.mean(dim=1, keepdim=True)

        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities

def get_mask_attention(attn, batch_size, sourceL, queryL, lamda=1):
    # attn --> (batch, sourceL, queryL)
    # positive attention
    mask_positive = attn.le(0)
    attn_pos = attn.masked_fill(mask_positive, torch.tensor(-1e9)) 
    attn_pos = torch.exp(attn_pos * lamda) 
    attn_pos = l1norm(attn_pos, 1) 
    attn_pos = attn_pos.view(batch_size, queryL, sourceL) # (batch_size, queryL, sourceL)

    return attn_pos

def evalrank(model_path, data_path=None, vocab_path=None, split="dev", fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint["args"]

    print("training epoch: ", checkpoint["epoch"])
    opt.workers = 0
    print(opt)
    if data_path is not None:
        opt.data_path = data_path
    if vocab_path is not None:
        opt.vocab_path = vocab_path

    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)

    try:
        # construct model
        net = SGRAF(opt)

        # load model state
        net.load_state_dict(checkpoint['net'])
    except Exception as e:
        opt.vocab_size = opt.vocab_size - 1
        # construct model
        net = SGRAF(opt)

        # load model state
        net.load_state_dict(checkpoint['net'])
    net = net.cuda()

    print('Loading dataset')
    if opt.data_name == "cc152k_precomp":
         images, captions, tags, image_ids, raw_captions = data_f30k.get_dataset(opt.data_path, opt.data_name, split, vocab, return_id_caps=True)
    else:
        images, captions, tags = data_f30k.get_dataset(opt.data_path, opt.data_name, split, vocab)
    data_loader = data_f30k.get_loader(images, captions, tags, split, opt.batch_size, opt.workers)

    print("Computing results...")
    with torch.no_grad():
        img_embs, cap_embs, cap_lens = train.encode_data(net, data_loader)
    print("Images: %d, Captions: %d"% (img_embs.shape[0] / per_captions, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation FIXME
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])

        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores(net, img_embs, cap_embs, cap_lens, opt, shard_size=1000)
        end = time.time()
        print("calculate similarity time:", end - start)

        # bi-directional retrieval
        # caption retrieval
        print('Single model evalution:')
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims, per_captions)
        print("Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(r1, r5, r10, medr, meanr))

        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims, per_captions)
        print("Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(r1i, r5i, r10i, medri, meanr))

    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            # 5fold split
            img_embs_shard = img_embs[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard = cap_embs[i * 5000 : (i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000 : (i + 1) * 5000]
            start = time.time()
            sims = shard_attn_scores(net, img_embs_shard, cap_embs_shard,cap_lens_shard, opt,shard_size=1000,)
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(img_embs_shard.shape[0], sims, per_captions=5, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard.shape[0], sims, per_captions=5, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        mean_i2t = (mean_metrics[0] + mean_metrics[1] + mean_metrics[2]) / 3
        print("Average i2t Recall: %.1f" % mean_i2t)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
        mean_t2i = (mean_metrics[5] + mean_metrics[6] + mean_metrics[7]) / 3
        print("Average t2i Recall: %.1f" % mean_t2i)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])

def i2t_RCL(images, captions, caplens, sims, npts=None, return_ranks=False, img_div=5):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # img_div = int(sims.shape[1] / sims.shape[0])

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        # Score
        rank = 1e20
        for i in range(img_div * index, img_div * index + img_div, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i_RCL(images, captions, caplens, sims, npts=None, return_ranks=False, img_div=5):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # img_div = int(sims.shape[1] / sims.shape[0])

    npts = sims.shape[0]
    ranks = np.zeros(img_div * npts)
    top1 = np.zeros(img_div * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(img_div):
            inds = np.argsort(sims[img_div * index + i])[::-1]
            ranks[img_div * index + i] = np.where(inds == index)[0][0]
            top1[img_div * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def evaluation(model_path=None, data_path=None, split='dev', fold5=False):
    module_names = ['SAF', 'SGR', 'SGRAF']

    sims_list = []
    for path in model_path:
        # load model and options
        checkpoint = torch.load(path)
        opt = checkpoint['args']
        save_epoch = checkpoint['epoch']
        print(opt)

        # load vocabulary used by the model
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
        vocab.add_word('<mask>')
        opt.vocab_size = len(vocab)

        try:
            # construct model
            model = SGRAF(opt)

            # load model state
            model.load_state_dict(checkpoint['net'])
        except Exception as e:
            opt.vocab_size = opt.vocab_size - 1
            # construct model
            model = SGRAF(opt)

            # load model state
            model.load_state_dict(checkpoint['net'])
        model = model.cuda()

        print('Loading dataset')
        if opt.data_name == "cc152k_precomp":
            captions, images, image_ids, raw_captions = data_f30k.get_dataset(
                opt.data_path, opt.data_name, split, vocab, return_id_caps=True
            )
        else:
            images, captions, tags = data_f30k.get_dataset(opt.data_path, opt.data_name, split, vocab)
        data_loader = data_f30k.get_loader(images, captions, tags, split, opt.batch_size, opt.workers)

        print("=> loaded checkpoint_epoch {}".format(save_epoch))

        print('Computing results...')
        img_embs, cap_embs, cap_lens = train.encode_data(model, data_loader)
        img_div = 1 if 'cc152k' in opt.data_name else 5# int(cap_embs.shape[0] / img_embs.shape[0])
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0] / img_div, cap_embs.shape[0]))

        sims_list.append([])
        if not fold5:
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), img_div)])
            start = time.time()
            sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=1000)
            end = time.time()
            print("calculate similarity time:", end-start)
            sims_list[-1].append(sims)
        else:
            for i in range(5):
                img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

                start = time.time()
                sims = shard_attn_scores(model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=1000)
                end = time.time()
                print("calculate similarity time:", end-start)
                sims_list[-1].append(sims)

    if len(sims_list) >= 2:
        sims_list_tmp = []
        for i in range(len(sims_list[0])):
            sim_tmp = 0
            for j in range(len(sims_list)):
                sim_tmp = sim_tmp + sims_list[j][i]
            sim_tmp /= len(sims_list)
            sims_list_tmp.append(sim_tmp)
        sims_list.append(sims_list_tmp)

    for j in range(len(sims_list)):
        if not fold5:
            sims = sims_list[j][0]
            # bi-directional retrieval
            r, rt = i2t_RCL(None, None, None, sims, return_ranks=True, img_div=img_div)
            ri, rti = t2i_RCL(None, None, None, sims, return_ranks=True, img_div=img_div)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("-----------------%s------------------" % module_names[j])
            print("rsum: %.1f" % rsum)
            print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
            print("---------------------------------------")
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            for i in range(5):
                sims = sims_list[j][i]

                r, rt0 = i2t_RCL(None, None, None, sims, return_ranks=True)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0 = t2i_RCL(None, None, None, sims, return_ranks=True)
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

                if i == 0:
                    rt, rti = rt0, rti0
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            print("-----------------%s------------------" % module_names[j])
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            print("rsum: %.1f" % (mean_metrics[10] * 6))
            print("Average i2t Recall: %.1f" % mean_metrics[11])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[:5])
            print("Average t2i Recall: %.1f" % mean_metrics[12])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[5:10])
            print("---------------------------------------")