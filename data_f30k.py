import pickle
import torch
import torch.utils.data as data
import numpy as np
import os
import nltk
from torch.utils.data import DataLoader
import copy
import csv


def collate_fn(data):
    if len(data[0]) == 6:   #only for correct_dataloader
        images, captions, tags, losses, ids, _labels = zip(*data)
    elif len(data[0]) == 5:
        images, captions, tags, ids, _labels = zip(*data)
    elif len(data[0]) == 4:
        images, captions, tags, ids = zip(*data)
    else:
        raise NotImplementedError("data length error!")

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths1 = [len(cap) for cap in captions]
    text = torch.zeros(len(captions), max(lengths1)).long()
    for i, cap in enumerate(captions):
        end = lengths1[i]
        text[i, :end] = cap[:end]

    lengths2 = [len(cap) for cap in tags]
    tag = torch.zeros(len(tags), max(lengths2)).long()
    for i, cap in enumerate(tags):
        end = lengths2[i]
        tag[i, :end] = cap[:end]

    if len(data[0]) == 6:
        return images, text, tag, lengths1, lengths2, losses, ids, _labels
    elif len(data[0]) == 5:
        return images, text, tag, lengths1, lengths2, ids, _labels
    elif len(data[0]) == 4:
        return images, text, tag, lengths1, lengths2, ids
    else:
        raise NotImplementedError("data length error!")

def get_dataset(data_path, data_name, data_split, vocab, return_id_caps=False):
    tag_loc = '.../coco_precomp/'+data_split+'_GITgenCaps.txt'
    data_path = os.path.join(data_path, data_name)

    # Captions
    captions = []
    tags = []
    if data_name == "cc152k_precomp":
        img_ids = []
        with open(os.path.join(data_path, "%s_caps.tsv" % data_split)) as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for line in tsvreader:
                captions.append(line[1].strip())
                img_ids.append(line[0])

    elif data_name in ["coco_precomp", "f30k_precomp"]:
        with open(os.path.join(data_path, "%s_caps.txt" % data_split), "r") as f:
            for line in f:
                captions.append(line.strip())

        with open(tag_loc, 'r') as f:#rb
            for line in f:
                tags.append(line.strip())        
    else:
        raise NotImplementedError("Unsupported dataset!")

    # caption tokens
    captions_token = []
    for index in range(len(captions)):
        caption = captions[index]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        captions_token.append(caption)

    # tags tokens
    tags_token = []
    for index in range(len(tags)):
        tag = tags[index]
        tokens = nltk.tokenize.word_tokenize(tag.lower())
        tag = []
        tag.append(vocab("<start>"))
        tag.extend([vocab(token) for token in tokens])
        tag.append(vocab("<end>"))
        tags_token.append(tag)

    # images
    images = np.load(os.path.join(data_path, "%s_ims.npy" % data_split))
    print("load {} / {} data: {} images, {} captions, {} tags".format(data_path, data_split, images.shape[0], len(captions), len(tags)))

    if return_id_caps:
        return images, captions_token, tags_token, img_ids, captions
    else:
        return images, captions_token, tags_token

class PrecompDataset(data.Dataset):
    def __init__(self, images, captions, tags, data_split, noise_ratio=0, noise_file=""):
        assert 0 <= noise_ratio < 1

        self.captions = captions
        self.images = images
        self.tags = tags
        self.noise_ratio = noise_ratio
        self.data_split = data_split

        self.length1 = len(self.captions)
        if self.images.shape[0] != self.length1:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "dev":
            self.length1 = 1000 * self.im_div

        # one image has five captions
        self.t2i_index = np.arange(0, self.length1) // self.im_div

        # Noisy label
        if data_split == "train" or data_split == "train_all":
            self._t2i_index = copy.deepcopy(self.t2i_index)

            if noise_ratio:
                if os.path.exists(noise_file):
                    print("=> load noisy index from {}".format(noise_file))
                    print('Noisy rate: %g' % noise_ratio)
                    self.t2i_index = np.load(noise_file)
                else:
                    idx = np.arange(self.length1)
                    np.random.shuffle(idx)
                    noise_length = int(noise_ratio * self.length1)

                    shuffle_index = self.t2i_index[idx[:noise_length]]
                    np.random.shuffle(shuffle_index)
                    self.t2i_index[idx[:noise_length]] = shuffle_index

                    np.save(noise_file, self.t2i_index)
                    print("=> save noisy index to {}".format(noise_file))

            # save clean labels
            self._labels = np.ones((self.length1), dtype="int")
            self._labels[self._t2i_index != self.t2i_index] = 0

        print("{} data has a size of {}".format(data_split, self.length1))

    def __getitem__(self, index):

        image = torch.Tensor(self.images[self.t2i_index[index]])
        text = np.array(self.captions[index])
        text = torch.Tensor(text)
        tag = np.array(self.tags[self.t2i_index[index] * self.im_div + (index % self.im_div)])
        tag = torch.Tensor(tag)

        if self.data_split == "train_all":
            return image, text, tag, index, self._labels[index] 
        else:
            return image, text, tag, index

    def __len__(self):
        return self.length1

def get_loader(images, captions, tags, data_split, batch_size, workers, noise_ratio=0, noise_file="", samper_seq = None):
    if data_split == "train":
        dset = PrecompDataset(images, captions, tags, "train", noise_ratio, noise_file)
        data_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False if samper_seq else True, pin_memory=True, collate_fn=collate_fn, num_workers=workers,)

    elif data_split == "train_all":
        dset = PrecompDataset(images, captions, tags, "train_all", noise_ratio, noise_file)
        data_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False if samper_seq else True, pin_memory=True, collate_fn=collate_fn, num_workers=workers,)

    elif data_split == "dev":
        dset = PrecompDataset(images, captions, tags, data_split)
        data_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, pin_memory=True,collate_fn=collate_fn,num_workers=workers,)

    elif data_split in ["test", "testall", "test5k"]:
        dset = PrecompDataset(images, captions, tags, data_split)
        data_loader = DataLoader(dataset=dset,batch_size=batch_size,shuffle=False, pin_memory=True, collate_fn=collate_fn, num_workers=workers,)    
    else:
        raise NotImplementedError("Not support data split!")

    return data_loader


class PrecompDataset_split(data.Dataset):
    def __init__(self, images, captions, tags,  losses, noise_ratio=0, noise_file="", mode="", pred=[]):
        assert 0 <= noise_ratio < 1

        self.captions = captions
        self.images = images
        self.tags = tags
        self.losses = losses
        self.noise_ratio = noise_ratio
        self.mode = mode

        self.length1 = len(self.captions)

        if self.images.shape[0] != self.length1:
            self.im_div = 5
        else:
            self.im_div = 1
        
        self.t2i_index = np.arange(0, self.length1) // self.im_div

        # Noisy label
        split_idx = None
        self._t2i_index = copy.deepcopy(self.t2i_index) 

        if noise_ratio:
            if os.path.exists(noise_file):
                print("=> load noisy index from {}".format(noise_file))
                self.t2i_index = np.load(noise_file) 
            else:
                idx = np.arange(self.length1)
                np.random.shuffle(idx) 
                noise_length = int(noise_ratio * self.length1) 

                shuffle_index = self.t2i_index[idx[:noise_length]] 
                np.random.shuffle(shuffle_index) 
                self.t2i_index[idx[:noise_length]] = shuffle_index 

                np.save(noise_file, self.t2i_index)
                print("=> save noisy index to {}".format(noise_file))

        self._labels = np.ones((self.length1), dtype="int") 
        self._labels[self._t2i_index != self.t2i_index] = 0

        if self.mode == "labeled": 
            split_idx = pred.nonzero()[0] 
        elif self.mode == "unlabeled": 
            split_idx = (1 - pred).nonzero()[0]

        if split_idx is not None:
            self.captions = [self.captions[i] for i in split_idx] 
            self.t2i_index = [self.t2i_index[i] for i in split_idx]
            self._t2i_index = [self._t2i_index[i] for i in split_idx] #clean
            self._labels = [self._labels[i] for i in split_idx]
            self.length1 = len(self.captions)

        print("{} data has a size of {}".format(self.mode, self.length1))

    def __getitem__(self, index):

        image = torch.Tensor(self.images[self.t2i_index[index]])
        text = torch.Tensor(self.captions[index])
        tag = torch.Tensor(self.tags[self.t2i_index[index] * self.im_div + (index % self.im_div)])

        loss = self.losses[index]

        if self.mode == "labeled":
            return (image, text, tag, loss, index, self._labels[index])
        elif self.mode == "unlabeled":
            return image, text, tag, index, self._labels[index]
        else:
            raise NotImplementedError("Not support data mode!")

    def __len__(self):
        return self.length1


def get_loader_split(images, captions, tags, losses, batch_size, workers, noise_ratio=0, noise_file="", pred=[],):

    dset_c = PrecompDataset_split(images, captions, tags,  losses, noise_ratio, noise_file,  mode="labeled", pred=pred)

    dset_n = PrecompDataset_split(images, captions, tags,  losses, noise_ratio, noise_file, mode="unlabeled", pred=pred)

    data_loader_c = DataLoader(dataset=dset_c, batch_size=batch_size, shuffle=True, pin_memory=True,
        collate_fn=collate_fn, num_workers=workers, drop_last=True)

    data_loader_n = DataLoader(dataset=dset_n, batch_size=batch_size, shuffle=True, pin_memory=True,
        collate_fn=collate_fn, num_workers=workers, drop_last=True)

    return data_loader_c, data_loader_n





