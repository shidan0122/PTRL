import os

import numpy as np 
import torch 
import time
import argparse
import data_f30k
from model import SGRAF
from train import Train, validate
from vocab import Vocabulary, deserialize_vocab
from torch.nn.utils.clip_grad import clip_grad_norm_
from tensorboardX import SummaryWriter
from utils import (AverageMeter, ProgressMeter, save_checkpoint, init_seeds, save_config)
from evaluation import i2t, t2i, LogCollector, logging_func, shard_attn_scores, evalrank

def warmup(opt, warm_trainloader, net, optimizer):
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(warm_trainloader), [batch_time, data_time, losses], prefix="Warmup Step")
    end = time.time()

    net.train()
    for iteration, (images, captions, tags, lengths1, lengths2, _) in enumerate(warm_trainloader):
        if images.size(0) == 1:
            break
        images, captions, tags = images.cuda(), captions.cuda(), tags.cuda()

        optimizer.zero_grad()

        loss1 = net(images, tags, lengths2, method = 'RCE')
        loss2 = net(images, captions, lengths1, method = 'RCE')
        loss = loss1 + loss2
        
        loss.backward()
        if opt.grad_clip > 0:
            clip_grad_norm_(net.parameters(), opt.grad_clip)

        optimizer.step()
        losses.update(loss.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % opt.log_step == 0:
            progress.display(iteration)

def split_data_by_loss(args, all_trainloader, net, images_train, captions_train, tags_train, epoch):
    net.eval()

    with torch.no_grad():
        data_num = len(all_trainloader.dataset)
        loss_all_captions = torch.zeros((data_num, 1)).cuda()
        loss_all_tags = torch.zeros((data_num, 1)).cuda()
        labels_all = torch.zeros((data_num), dtype = torch.long)

        for iteration, (images, captions, tags, lengths1, lengths2, ids, labels) in enumerate(all_trainloader):
            images, captions, tags = images.cuda(), captions.cuda(), tags.cuda()
            labels = torch.tensor(labels)

            sim_captions, _ = net.forward_all(images, captions, lengths1)
            sim_tags, _ = net.forward_all(images, tags, lengths2)
            sim_captions_diag = torch.diag(sim_captions)
            sim_tags_diag = torch.diag(sim_tags)

            loss_all_captions[ids] = sim_captions_diag.unsqueeze(1)
            loss_all_tags[ids] = sim_tags_diag.unsqueeze(1)
            labels_all[ids] = labels

            del sim_captions, sim_tags, sim_captions_diag, sim_tags_diag

        labels_all = labels_all.numpy()
        loss_all_captions = loss_all_captions.cpu().numpy()
        loss_all_tags = loss_all_tags.cpu().numpy()

        delta = 0.0
        pred = (loss_all_captions >= loss_all_tags + delta) 
        pred = pred.squeeze()
        correct_labels = labels_all[pred]
        print('Correct data acc:', sum(correct_labels) / len(correct_labels))
        print('Total data acc:', sum(labels_all == pred) / len(labels_all))

    correct_trainloader, noisy_trainloader = data_f30k.get_loader_split(images_train, captions_train, tags_train, loss_all_captions, 
        args.batch_size, args.workers, noise_ratio=args.noise_ratio, noise_file=args.noise_file, pred=pred)    
        
    del loss_all_captions, loss_all_tags, labels_all
    
    return correct_trainloader, noisy_trainloader

def main():
    current_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    # Data Settings
    parser.add_argument("--data_path", default=" ", help="path to datasets")
    parser.add_argument('--data_name', default='coco_precomp', help='{coco_precomp,f30k_precomp}')
    parser.add_argument('--vocab_path', default='.../vocab/', help='Path to saved vocabulary json files.')

    # Noise Settings                
    parser.add_argument('--noise_ratio', default=0.2, type=float, help='Noise rate.')
    parser.add_argument("--noise_file", 
                        default=".../noise_index/coco_precomp_0.2.npy", help="noise_file")

    # Training Settings
    parser.add_argument("--batch_size", default=128, type=int, help="Size of a training mini-batch.")
    parser.add_argument("--num_epochs", default=30, type=int, help="Number of training epochs.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Initial learning rate.")#0.0005 0.001
    parser.add_argument('--lr_update', default=20, type=int, help='Number of epochs to update the learning rate.')#20,30,15
    parser.add_argument('--grad_clip', default=2.0, type=float, help='Gradient clipping threshold.')#5
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')

    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')

    # Runing Settings
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")#1
    parser.add_argument("--warmup_model_path", default="", help="warm up models")

    parser.add_argument("--output_dir", default=os.path.join("output", current_time), help="Output dir.")
    parser.add_argument('--log_step', default=200, type=int, help='Number of steps to print and record the log.')#200
    parser.add_argument("--tau", default=0.05, type=float, help="temperature coefficient")

    # Model Settings
    parser.add_argument('--img_dim', default=2048, type=int, help='Dimensionality of the image embedding.')
    parser.add_argument('--txt_dim', default=300, type=int, help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_dim', default=1024, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument("--sim_dim", default=256, type=int, help="Dimensionality of the sim embedding.")
    parser.add_argument('--num_layers', default=1, type=int,help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',help='Do not normalize the text embeddings.')

    parser.add_argument("--module_name", default="SGR", type=str, help="SGR, SAF")
    parser.add_argument("--sgr_step", default=3, type=int, help="Step of the SGR.")

    parser.add_argument('--param1', default=0.5, type=float, help='parameter of loss_c1.')
    parser.add_argument('--param2', default=0.5, type=float, help='parameter of loss_n.')

    args = parser.parse_args()
    print("\n*-------- Experiment Config --------*")
    print(args)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not args.noise_file:
        args.noise_file = os.path.join(args.output_dir, args.data_name + "_" + str(args.noise_ratio) + ".npy")
    if args.data_name == "cc152k_precomp":
        args.noise_ratio = 0
        args.noise_file = ""

    save_config(args, os.path.join(args.output_dir, "config.json"))
    writer = SummaryWriter(os.path.join('runs', args.output_dir))

    init_seeds(args.seed)
    print("********load and process dataset ...")
    vocab = deserialize_vocab(os.path.join(args.vocab_path, '%s_vocab.json' % args.data_name))
    #vocab.add_word('<mask>')
    args.vocab_size = len(vocab)

    images_train, captions_train, tags_train = data_f30k.get_dataset(args.data_path, args.data_name, "train", vocab)
    images_dev, captions_dev, tags_dev = data_f30k.get_dataset(args.data_path, args.data_name, "dev", vocab)

    all_trainloader = data_f30k.get_loader(images_train, captions_train, tags_train, "train_all", args.batch_size, args.workers, 
        noise_ratio = args.noise_ratio, noise_file = args.noise_file, samper_seq = False)
    val_loader = data_f30k.get_loader(images_dev, captions_dev, tags_dev, "dev", args.batch_size, args.workers)

    net = SGRAF(args).cuda()

    if args.warmup_model_path:
        if os.path.isfile(args.warmup_model_path):
            print('Load warm up model')
            checkpoint = torch.load(args.warmup_model_path)
            net.load_state_dict(checkpoint["net"], strict=False)
            print("=> load warmup checkpoint '{}' (epoch {})".format(args.warmup_model_path, checkpoint["epoch"]))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(args.warmup_model_path))

    best_rsum = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate) 

    for epoch in range(args.num_epochs):
        print('Epoch', epoch+1, '/', args.num_epochs)
        print("********Split dataset ...")
        correct_trainloader, noisy_trainloader = split_data_by_loss(args,all_trainloader,net,images_train,captions_train,tags_train,epoch)

        print("\n ********Model training ...")
        Train(args, correct_trainloader, noisy_trainloader, net, optimizer, epoch)

        print("\n ********Validattion ...")
        r1, r5, r10, r1i, r5i, r10i = validate(args, val_loader, [net])
        rsum = r1 +  r5 + r10 + r1i + r5i + r10i
        writer.add_scalar('Image to Text R1', r1, global_step=epoch, walltime=None)
        writer.add_scalar('Image to Text R5', r5, global_step=epoch, walltime=None)
        writer.add_scalar('Image to Text R10', r10, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R1', r1i, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R5', r5i, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R10', r10i, global_step=epoch, walltime=None)
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint({"epoch": epoch, "net": net.state_dict(), "best_rsum": best_rsum, "args": args,},
                is_best, filename="checkpoint_{}_0830.pth.tar".format(epoch), prefix=args.output_dir + "/",)

    # test
    print("\n*-------- Testing --------*")
    if args.data_name == "coco_precomp":
        print("5 fold validation")
        evalrank(os.path.join(args.output_dir, "model_best.pth.tar"), split="testall", fold5=True,)
        print("full validation")
        evalrank(os.path.join(args.output_dir, "model_best.pth.tar"), split="testall")
    else:
        evalrank(os.path.join(args.output_dir, "model_best.pth.tar"), split="test")

if __name__ == "__main__":
    print("\n*-------- Training & Testing --------*")
    main()