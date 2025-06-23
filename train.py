import time
import logging
from evaluation import i2t, t2i, LogCollector, logging_func, shard_attn_scores, evalrank
import torch
import numpy as np
import os
from utils import (AverageMeter, ProgressMeter, save_checkpoint, init_seeds, save_config)
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F

def Train(args, correct_trainloader, noisy_trainloader, net, optimizer, epoch):
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    progress = ProgressMeter(len(correct_trainloader), [batch_time, losses], prefix="Training Step")

    learning_rate = args.learning_rate * (0.1 ** (epoch // args.lr_update))
    for group in optimizer.param_groups:
        group['lr'] = learning_rate

    print('\n Training...')
    end = time.time()

    noisy_train_iter = iter(noisy_trainloader)
    for iteration, (images_c, captions_c, tags_c, lengths1_c, lengths2_c, losses_c, _, _) in enumerate(correct_trainloader):
        images_c, captions_c, tags_c, losses_c = images_c.cuda(), captions_c.cuda(), tags_c.cuda(), np.array(losses_c).squeeze()
        try:
            images_n, captions_n, tags_n, lengths1_n, lengths2_n, _, _ = next(noisy_train_iter)
        except:
            noisy_train_iter = iter(noisy_trainloader)
            images_n, captions_n, tags_n, lengths1_n, lengths2_n, _, _ = next(noisy_train_iter)

        images_n, captions_n, tags_n = images_n.cuda(), captions_n.cuda(), tags_n.cuda()
        
        net.train()
        optimizer.zero_grad()

        loss_c = net(images_c, captions_c, lengths1_c, method = 'RCE')
        loss_c1 = net(images_c, tags_c, lengths2_c, method = 'RCE')
        loss_n = net(images_n, tags_n, lengths2_n, method = 'RCE')
        
        loss = loss_c + args.param1*loss_c1 + args.param2*loss_n

        loss.backward()
        if args.grad_clip > 0:
            clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()
        losses.update(loss.item(), images_c.size(0))

        #print(f"loss: {loss.item()}, loss_c: {loss_c.item()}, loss_n: {loss_n.item()}")

        batch_time.update(time.time() - end)
        end = time.time()

        torch.cuda.empty_cache()

        if iteration % args.log_step == 0:
            progress.display(iteration)

def validate(args, val_loader, net= []):
    if args.data_name == "cc152k_precomp":
        per_captions = 1
    elif args.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5
    sims_mean = 0
    count = 0

    for ind in range(len(net)):
        count += 1
        print("Encoding with model {}".format(ind))

    img_embs, cap_embs, cap_lens = encode_data(net[ind], val_loader, args.log_step)
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])

    start = time.time()
    print("Computing similarity from model {}".format(ind))

    sims_mean += shard_attn_scores(net[ind], img_embs, cap_embs, cap_lens, args, shard_size=1000)#100 128

    end = time.time()
    print("Calculate similarity time with model {}: {:.2f} s".format(ind, end - start))

    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    print("Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    print("Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return r1 ,r5 ,r10 , r1i ,r5i ,r10i

def encode_data(model, data_loader, log_step=10, logging=print):
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(data_loader), [batch_time, data_time], prefix="Encode")

    model.eval()

    img_embs = None
    cap_lens = None

    max_n_word = 0
    for i, (images, captions, _, lengths1, lengths2, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths1))

    image_ids = []
    end = time.time()
    
    for i, (images, captions, _, lengths1, lengths2, ids) in enumerate(data_loader):
        data_time.update(time.time() - end)
        images, captions = images.cuda(), captions.cuda()

        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths1)

        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths1), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            progress.display(i)

        torch.cuda.empty_cache()
        del images, captions

    return img_embs, cap_embs, cap_lens
