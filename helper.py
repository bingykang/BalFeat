import argparse
import os
import random
import shutil
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

import moco.loader
import models
import loader
import tensorflow as tf
gfile = tf.io.gfile

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_datadir(args):
    dataset = args.dataset
    if dataset == 'imagenet':
        # return 'hdfs://haruna/home/byte_arnold_lq/user/yibairen.byron/imagenet', 1000
        return '/opt/tiger/bykang/imagenet', 1000
    elif dataset == 'places365':
        return 'hdfs://haruna/home/byte_arnold_lq_vc/user/bykang/places365_standard', 365
    elif dataset == 'imagenetc':
        return '/root/imagenet_c', 1000
    elif dataset == 'inat18':
        args.trainset = 'data/iNaturalist18_train.txt'
        args.valset = 'data/iNaturalist18_val.txt'
        return '/opt/tiger/bykang/inat18', 8142
    else:
        raise ValueError('Dataset: {} not found'.format(dataset))

def get_default_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # parser.add_argument('data', metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('--data', metavar='DIR',
                        default='hdfs://haruna/home/byte_arnold_lq/user/yibairen.byron/imagenet',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true',
    #                     help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--dist', action='store_true', help='quick setup '
                        'single-machine distributed training')

    # Learning rate stuff 
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--lrend', default=0, type=float,
                        help='end lr for cosine lr schedule')
    
    # optimizer (mainly for vit training)
    parser.add_argument('--opt', type=str, default='sgd', 
                        help='optimizer used for training')
    # AMD speedup
    parser.add_argument('--amp', action='store_true',
                        help='use amd for speedup')
    return parser

def add_data_arguments(parser):
    # option for dataset
    parser.add_argument('--dataset', default='imagenet', type=str,
                        choices=['imagenet', 'places365', 'inat18'],
                        help='path to subset txt file for training')
    parser.add_argument('--num_classes', default=1000, type=int,
                        help='number of classes')
    # options for sampled datasets
    parser.add_argument('--trainset', default=None, type=str,  metavar='TXT FILE',
                        help='path to subset txt file for training')
    parser.add_argument('--valset', default=None, type=str,  metavar='TXT FILE',
                        help='path to subset txt file for evaluation')
    parser.add_argument('--clsset', default=None, type=str,  metavar='TXT FILE',
                        help='path to subset txt file for cls splits')
    parser.add_argument('--balanced', action='store_true',
                        help='use balanced sampling for training')
    parser.add_argument('--bal_repeat', default=1, type=int,
                        help='the repeat number for balanced sampling')
    # options for KV data loader
    parser.add_argument('--num_readers', default=32, type=int,
                        help='number of readers for KVreader')
    parser.add_argument('--logdir', default='', type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--name', default='', type=str,
                        help='name')
    return parser


def get_remote_file(remote_file):
    if isinstance(remote_file, str) and remote_file.startswith('hdfs'):
        local_file = os.path.join('/tmp', remote_file.split('/')[-1])
        if os.path.exists(local_file):
            os.remove(local_file)
        print('=> Copying {} to {}'.format(remote_file, local_file))
        gfile.copy(remote_file, local_file)
        return local_file
    else:
        return remote_file

def main(main_worker, parser):
    args = parser.parse_args()

    # Setup dataset issue
    args.data, args.num_classes = get_datadir(args)
    if 'rotation' in args and args.rotation:
        args.num_classes = 4
    if args.clsset is not None:
        args.num_classes = 500

    # Quick setup of single-machine multi-processing training 
    if args.dist:
        args.multiprocessing_distributed = True
        args.dist_backend = 'nccl'
        args.world_size = 1
        args.rank = 0
        if not args.dist_url.startswith('tcp://127.0.0.1'):
            args.dist_url = 'tcp://127.0.0.1:10001'
    # Take care of hdfs files
    args.pretrained = get_remote_file(args.pretrained)
    args.resume = get_remote_file(args.resume)
    # Balanced sampling is only used for subset imagenet
    if not args.evaluate and args.balanced:
        assert args.trainset is not None

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        scaling = 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        if args.lrend > 0:
            lr = args.lrend + scaling * (args.lr - args.lrend)
        else:
            lr *= scaling
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    elif 'schedule' in args and args.schedule:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    else:
        lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def reset_default_value(parser, dest, default):
    for action in parser._actions:
        if action.dest == dest:
            action.default = default
            return
    else:
        raise AssertionError('argument {} not found'.format(dest))


def save_checkpoint(state, is_best, filename=''):
    ckptname = filename + '-ckpt.pth.tar'
    torch.save(state, ckptname)
    if is_best:
        bestname = filename if filename else 'model_best'
        shutil.copyfile(ckptname, bestname + '.pth.tar')


def get_dataloader(args, dataroot, transform, is_val=False, drop_last=False):
    subset = args.valset if is_val else args.trainset
    if dataroot.startswith('hdfs'):
        if 'rotation' in args and args.rotation:
            DS = loader.KVRotationDataset
        else:
            DS = loader.KVDataset

        dataset = DS(dataroot, args.num_readers, transform, subset=subset,
                     clsset=args.clsset)

        if args.distributed:
            num_replicas, rank = None, None
        else:   
            num_replicas, rank = 1, 0
        
        # Balanced sampling is noly used for training
        if args.balanced and not is_val:
            data_sampler = loader.KVBalancedSampler(
                dataset, batch_size=args.batch_size, num_replicas=num_replicas,
                rank=rank, drop_last=drop_last, repeat=args.bal_repeat)
        else:
            data_sampler = loader.KVSampler(
                dataset, batch_size=args.batch_size, num_replicas=num_replicas,
                rank=rank, shuffle=not is_val, drop_last=drop_last)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=args.workers, pin_memory=True,
            sampler=data_sampler, worker_init_fn=loader.worker_init_fn)
    else:
        # assert subset is None, 'subset dataset is not implemented'
        # dataset = datasets.ImageFolder(dataroot, transform)
        DS = loader.Inaturalist if args.dataset == 'inat18' else loader.ImageFolder
        dataset = DS(dataroot, subset=subset, clsset=args.clsset,
                     transform=transform)

        if args.distributed:
            SAMPLER = loader.DistributedBalancedSampler if args.balanced else \
                      torch.utils.data.distributed.DistributedSampler
            data_sampler= SAMPLER(dataset)
            # data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            assert not args.balanced, 'balanced sampling is not implemented'
            data_sampler = None

        shuffle = False if is_val else (data_sampler is None)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.workers, pin_memory=True, sampler=data_sampler,
            drop_last=drop_last)
    
    return dataset, data_sampler, data_loader


def get_transforms(dataset=None, aug=None):
    "Get tranforms for based on datasets"
    if dataset == 'inat18':
        aug = None
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380],
                                         std=[0.195, 0.194, 0.192])
    else:
        # The default routine for ImageNet, Places
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # validation aug
    vt = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    # training aug
    if aug == 'moco':
        tt = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif aug == 'mocov2':
        tt = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif aug == 'lt':
        tt = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        tt = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return tt, vt
