import os
import dill
import numpy as np
from sklearn.utils import shuffle
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torchtext import data
from torchtext import datasets
from torchtext.data import Dataset

import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from medmnist.dataset import DermaMNIST

from warfit_learn import preprocessing
from warfit_learn import datasets as warfit_dataset
## for VOC
import cv2
from VOC.data_preprocessing import TrainAugmentation, TestTransform
from VOC.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors, MatchPrior
from VOC.misc import str2bool, Timer, freeze_net_layers, store_labels
import xml.etree.ElementTree as ET
## for imagenette 
from fastai.basics import untar_data
from fastai.vision.all import *

def get_imagenette_dataloader(
    bs=64,
    item_tfms=[RandomResizedCrop(size=128, min_scale=0.35), FlipItem(0.5)],
    batch_tfms=RandomErasing(p=0.9, max_count=3)
):
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=GrandparentSplitter(valid_name='val'),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    data_path = '/home/myang_20210409/data/imagenette2/'
    val_data_path = '/home/myang_20210409/data/imagenette2/val/'
    dloader = dblock.dataloaders(untar_data(URLs.IMAGENETTE_320), path=untar_data(URLs.IMAGENETTE_320), bs=bs, num_workers=0)
    return dloader.train, dloader.valid

## For seq2seq
# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open('data/seq2seq_data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    MAX_LENGTH = 10
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    SOS_token = 0
    EOS_token = 1
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorsFromPair(pair,input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

class eng_fra_dataset():
    def __init__(self, pairs, input_lang, output_lang):
        self.Xs = []
        self.Ys = []
        for p in pairs:
            self.Xs.append(tensorsFromPair(p, input_lang, output_lang)[0])
            self.Ys.append(tensorsFromPair(p,input_lang, output_lang)[1])
    def __getitem__(self, index):
        X, y = self.Xs[index], self.Ys[index]
        return X, y
    def __len__(self):
        return len(self.Xs)

def get_eng_fra_dataloader(BATCH_SIZE=1):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(input_lang.n_words, output_lang.n_words)
    pairs_size = len(pairs)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang) for i in range(78200)]
    test_pairs = [tensorsFromPair(pair, input_lang, output_lang) for pair in pairs]
    return training_pairs, test_pairs

    train_dataset = eng_fra_dataset(pairs[:int(pairs_size*0.8)],input_lang, output_lang)
    test_dataset = eng_fra_dataset(pairs[int(pairs_size*0.8):],input_lang, output_lang)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_dataset = eng_fra_dataset(pairs,input_lang, output_lang)
    all_loader = DataLoader(dataset=all_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # return train_loader, test_loader
    # return all_loader, all_loader
    return train_loader, train_loader

    

# https://zhuanlan.zhihu.com/p/64934558
class IMDBDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

def dump_examples(train, test, suffix = None):
    # save the examples
    train_path, test_path = os.path.join('./data/text', suffix+'_train'), os.path.join('./data/text', suffix+'_test')
    if not os.path.exists(train_path):
        with open(train_path, 'wb')as f:
            dill.dump(train.examples, f)
    if not os.path.exists(test_path):
        with open(test_path, 'wb')as f:
            dill.dump(test.examples, f)

def load_split_datasets(fields, suffix = None):
    # load the examples
    train_path, test_path = os.path.join('./data/text', suffix+'_train'), os.path.join('./data/text', suffix+'_test')
    with open(train_path, 'rb')as f:
        train_examples = dill.load(f)
    with open(test_path, 'rb')as f:
        test_examples = dill.load(f)

    # æ¢å¤æ•°æ®é›†
    train = IMDBDataset(examples=train_examples, fields=fields)
    test = IMDBDataset(examples=test_examples, fields=fields)
    return train, test

class IMDB():
    def  __init__(self, device = None):
        super(IMDB, self).__init__()
        self.dataset_name = 'IMDB'
        self.batch_size = 128
        self.vocabulary_size = 20000
        self.random_seed = 123
        # torch.manual_seed(self.random_seed)
    
    def get_dataloader(self, verbose = False):
        TEXT = data.Field(tokenize='spacy',tokenizer_language="en_core_web_sm") # include_lengths=True) # necessary for packed_padded_sequence
        LABEL = data.LabelField(dtype=torch.float)
        text_vocab_path = os.path.join('./data/text', self.dataset_name+'_text_vocab')
        label_vocab_path = os.path.join('./data/text', self.dataset_name+'_label_vocab')

        if os.path.exists(os.path.join('./data/text', self.dataset_name +'_train')) and os.path.exists(text_vocab_path):
            if verbose:
                print('load the examples...')
            fields = {'text': TEXT, 'label': LABEL}
            train_data, test_data = load_split_datasets(fields = fields, suffix = self.dataset_name)
            with open(text_vocab_path, 'rb')as f:
                TEXT.vocab = dill.load(f)
            with open(label_vocab_path, 'rb')as f:
                LABEL.vocab = dill.load(f)
        else:
            if verbose:
                print('generate the examples...')
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            dump_examples(train_data, test_data, suffix = self.dataset_name)
            TEXT.build_vocab(train_data, max_size=self.vocabulary_size)
            LABEL.build_vocab(train_data)
            with open(text_vocab_path, 'wb')as f:
                dill.dump(TEXT.vocab, f)
            with open(label_vocab_path, 'wb')as f:
                dill.dump(LABEL.vocab, f)
        if verbose:
            print(f'Num Train: {len(train_data)}')
            print(f'Num Test: {len(test_data)}')

            print(f'Vocabulary size: {len(TEXT.vocab)}')
            print(f'Number of classes: {len(LABEL.vocab)}')
        
        train_loader, test_loader = data.BucketIterator.splits(
            (train_data, test_data), 
            batch_size=self.batch_size,
            sort_within_batch=True, # necessary for packed_padded_sequence
            )
        return train_loader, test_loader

## The dataset SPEECHCOMMANDS is a torch.utils.data.Dataset version of the dataset.
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

# def label_to_index(word):
#     # Return the position of the word in labels
#     return torch.tensor(labels.index(word))

# def index_to_label(index):
#     # Return the word corresponding to the index in labels
#     # This is the inverse of label_to_index
#     return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    # [Bacth_size, s, hz] -> [Bacth_size, pad(hz), channels] -> [Bacth_size, channels, hz], channels.value is between (-1, 1)
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def get_SC_dataloader(batch_size = 256, device = None):
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

    def collate_fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        tensors, targets = [], []
        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            ## The Sample frequency is still 16000 before transform (8000)
            ## The operation of transform is much slower in CPU the GPU
            # tensors += [transform(waveform)]
            tensors += [waveform]
            targets += [torch.tensor(labels.index(label))]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    ## CUDA
    num_workers = 0 # 1
    pin_memory = True
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader

## The dataset SPEECHCOMMANDS is a torch.utils.data.Dataset version of the dataset.
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    # [Bacth_size, s, hz] -> [Bacth_size, pad(hz), channels] -> [Bacth_size, channels, hz], channels.value is between (-1, 1)
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def get_SC_dataloader(batch_size = 256, new_sample_rate = None):
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    def collate_fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        tensors, targets = [], []
        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            ## The Sample fRate is 16000 before transform.
            ## The operation of transform is much slower in CPU the GPU.
            if new_sample_rate is not None:
                transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
                tensors += [transform(waveform)]
            else:
                tensors += [waveform]
            targets += [torch.tensor(labels.index(label))]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    ## not CUDA
    num_workers = 0
    pin_memory = False
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader

## export 
def get_mnist_dataloader_32(batch_size = 32, img_size = 32, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] , [0.5])])
    #Defining the transforms to applied on the image ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_mnist_dataloader_28(batch_size = 32, img_size = 28, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] , [0.5])])
    #Defining the transforms to applied on the image ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_cifar10_dataloader(batch_size = 64, **kwargs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0) # 2
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0) # 2
    return train_loader, test_loader

def get_cifar10_dataloader_gan():
    transform_train = transforms.Compose([
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0) # 2
    return train_loader,None

def get_dermamnist_dataloader(BATCH_SIZE=100, data_size = 32): # 28
    BATCH_SIZE=16
    transform = transforms.Compose([
        transforms.Resize(data_size),
        transforms.CenterCrop(data_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = DermaMNIST('data', split='train', transform=transform) # transform is only applied to images, not labels
    val_dataset = DermaMNIST('data', split='val', transform=transform)
    test_dataset = DermaMNIST('data', split='test', transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader

def get_imdb_dataloader(**kwargs):
    imdb = IMDB()
    train_loader, test_loader = imdb.get_dataloader(**kwargs)
    return train_loader, test_loader

def get_speechcommand_dataloader(**kwargs):
    return get_SC_dataloader(**kwargs)

def get_gtsrb_dataloader(batch_size=128, data_size = 32, **kwargs):
    gtsrb_data_transforms = transforms.Compose([
        transforms.Resize([data_size, data_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]),
    ])
    train_data_path = "~/data/GTSRB/train"
    train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform = gtsrb_data_transforms)
    test_data_path = "~/data/GTSRB/val"
    test_data = torchvision.datasets.ImageFolder(root = test_data_path, transform = gtsrb_data_transforms)
    train_loader = DataLoader(train_data, shuffle=True, batch_size = batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size = batch_size)
    return train_loader,test_loader

class WarfitDataset():  ## In python 3.9 need to remove the dataset
    def __init__(self, df, target_column='Therapeutic Dose of Warfarin'):
        self.ys = df[target_column].values.astype(np.float32)
        self.Xs = df.drop([target_column], axis=1).values.astype(np.float32)
    def __getitem__(self, index):
        X, y = self.Xs[index], self.ys[index]
        return X, y
    def __len__(self):
        return self.Xs.shape[0]

def get_warfit_dataloader(BATCH_SIZE=32):
    raw_iwpc = warfit_dataset.load_iwpc()
    df = preprocessing.prepare_iwpc(raw_iwpc)
    df['Height (cm)'] = df['Height (cm)'] / 100
    df['Weight (kg)'] = df['Weight (kg)'] / 100
    df = shuffle(df)
    
    pos = int(len(df) * 0.1)
    train_df = df[0 : 7 * pos]
    val_df = df[7 * pos : 8 * pos]
    test_df = df[8 * pos :]
    # print(len(train_df), len(val_df), len(test_df))
    
    train_dataset = WarfitDataset(train_df)
    val_dataset = WarfitDataset(val_df)
    test_dataset = WarfitDataset(test_df)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

class TSDataset():
    def __init__(self, Xs, ys):
        self.Xs = Xs
        self.ys = ys
    def __getitem__(self, index):
        X, y = self.Xs[index], self.ys[index]
        return X, y
    def __len__(self):
        return self.Xs.shape[0]

def get_ts_dataloader(dataset_name = 'UWaveGestureLibrary_3dim', BATCH_SIZE=64):
    data_dir = 'data/ts_data/data_{}.pkl'.format(dataset_name)
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    train_dataset = TSDataset(data['train_x'], data['train_l'])
    test_dataset = TSDataset(np.array(data['test_x']), np.array(data['test_l']))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
            From https://github.com/tranleanh/mobilenets-ssd-pytorch
        """
        self.transform = transform
        self.target_transform = target_transform

        if is_test:
            image_sets_file = "/home/myang_20210409/data/VOCdevkit/test/VOC2007/ImageSets/Main/test.txt"
        else:
            image_sets_file = "/home/myang_20210409/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"

        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        # if the labels file exists, read in the class names
        # label_file_name = self.root + "labels.txt"
        label_file_name = ""
        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()
            # classes should be a comma separated list
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            #logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            # print("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, [boxes, labels]

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        try:
            annotation_file = "/home/myang_20210409/data/VOCdevkit/VOC2007/" + f"Annotations/{image_id}.xml"
            objects = ET.parse(annotation_file).findall("object")
        except:
            annotation_file = "/home/myang_20210409/data/VOCdevkit/test/VOC2007/" + f"Annotations/{image_id}.xml"
            objects = ET.parse(annotation_file).findall("object")
        # objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')
                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        try:
            image_file = "/home/myang_20210409/data/VOCdevkit/VOC2007/" + f"JPEGImages/{image_id}.jpg"
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image_file = "/home/myang_20210409/data/VOCdevkit/test/VOC2007/" + f"JPEGImages/{image_id}.jpg"
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

def get_VOC_dataloader(batch_size = 24):
    print("____________________ batch size is 24 ____________________") # 24
    image_size = 300
    image_mean = np.array([127, 127, 127])  # RGB layout
    image_std = 128.0
    iou_threshold = 0.45
    center_variance = 0.1
    size_variance = 0.2
    specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
    ]
    priors = generate_ssd_priors(specs, image_size)
    train_transform = TrainAugmentation(image_size, image_mean, image_std)
    target_transform = MatchPrior(priors, center_variance,
                                  size_variance, 0.5)

    test_transform = TestTransform(image_size, image_mean, image_std)
    dataset_path = '/home/myang_20210409/data/VOCdevkit/VOC2007/'
    valid_dataset_path = '/home/myang_20210409/data/VOCdevkit/test/VOC2007/'
    train_dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
    label_file = os.path.join('data/', "voc-model-labels.txt")
    store_labels(label_file, train_dataset.class_names)
    num_classes = len(train_dataset.class_names)
    # print(num_classes) # 21
    # logging.info(f"Stored labels into file {label_file}.")
    # print("VOC Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=0,
                              shuffle=True)
    # logging.info("Prepare Validation datasets.")
    val_dataset = VOCDataset(valid_dataset_path, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    # print("VOC validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, batch_size,
                            num_workers=0,
                            shuffle=False)
    return train_loader, val_loader

class PairVoice(torch.utils.data.Dataset):
    def __init__(self,taskModel,pair_size):
        super(PairVoice, self).__init__()
        taskModel.pair_size = pair_size
        taskModel.num_pair = pair_size*taskModel.batch_size
        self.num_pair = taskModel.num_pair
        taskModel.fix_noise = torch.randn(taskModel.num_pair,100,1)
        self.fix_noise = taskModel.fix_noise
        # matching pair (nosie,tarining-set image)
        for i,(data,_) in enumerate(taskModel.train_loader):
            if i==0:
                voice_pair = data
            else:
                voice_pair = torch.cat((voice_pair,data),dim=0)
            if i+1>=pair_size:
                break
        taskModel.image_pair = voice_pair
        self.voice_pair = taskModel.image_pair

    def __getitem__(self, index):
        return self.fix_noise[index],self.voice_pair[index]
    
    def __len__(self):
        return self.num_pair

class PairImage(torch.utils.data.Dataset):
    def __init__(self,taskModel,pair_size):
        super(PairImage, self).__init__()
        taskModel.pair_size = pair_size
        taskModel.num_pair = pair_size*taskModel.batch_size
        self.num_pair = taskModel.num_pair
        taskModel.fix_noise = torch.randn(taskModel.num_pair,100,1,1)
        self.fix_noise = taskModel.fix_noise
        # matching pair (nosie,tarining-set image)
        for i,(data,_) in enumerate(taskModel.train_loader):
            if i==0:
                image_pair = data
            else:
                image_pair = torch.cat((image_pair,data),dim=0)
            if i+1>=pair_size:
                break
        taskModel.image_pair = image_pair
        self.image_pair = taskModel.image_pair

    def __getitem__(self, index):
        return self.fix_noise[index],self.image_pair[index]
    
    def __len__(self):
        return self.num_pair
        
if __name__ == '__main__':
    # print(untar_data(URLs.IMAGENETTE_320))
    train, valid = get_warfit_dataloader()
    print(len(train))
    print(len(valid))
    for x in train:
        print(type(x[0]))
        break