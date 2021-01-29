import argparse
import os
import random
import shutil
import time
import warnings
import builtins

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

import models
import util
from helper import *
from util import AverageMeter, ProgressMeter, accuracy

parser = get_default_parser()
parser = add_data_arguments(parser)
# reset_default_value(parser, 'workers', 32)
parser.add_argument('--rotation', action='store_true',
                    help='rotation')
parser.add_argument('--schedule', default=None, nargs='*', type=int,	
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--taunorm', action='store_true',
                    help='use taunorm for evaluation')
parser.add_argument('--save_acc', action='store_true',
                    help='save perclass accuracies')
parser.add_argument('--bp', action='store_true',
                    help='backpropogate gradient to backbone network')
parser.add_argument('--valdir', default=None, type=str,
                    help='image folder for evaluation')
parser.add_argument('--ltaug', action='store_true',
                    help='use longtail data augmentation')
parser.add_argument('--eval_mode', action='store_true',
                    help='use eval mode for learning classifier')
best_acc1 = 0


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        # Freezing backbone 
        if not args.bp:
            print('=> freezing backbone networks')
            for name, param in model.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            # init the fc layer
            # model.fc.weight.data.normal_(mean=0.0, std=0.01)
            # model.fc.bias.data.zero_()

        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            if 'state_dict_best' in checkpoint:
                checkpoint = {'state_dict': checkpoint['state_dict_best']['feat_model']}

            if 'OrderedDict' == checkpoint.__class__.__name__:
                for k in list(checkpoint.keys()):
                    checkpoint['module.' + k] = checkpoint[k]
                    del checkpoint[k]
                checkpoint = {'state_dict': checkpoint}

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k.replace("module.encoder_q.",'')] = state_dict[k]
                elif k.startswith('module.cls_fc'):
                    state_dict[k.replace("module.cls_fc",'fc')] = state_dict[k]
                elif k.startswith('module.'):
                    if args.dataset == 'places365' and \
                        k in ['module.fc.weight', 'module.fc.bias']:
                        continue
                    state_dict[k.replace("module.",'')] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    #TODO hyper for vit
    # optimizer AdamW
    # lr: 1e-3 -> 1e-5
    # weight_decay: 5e-2
    # cos lr
    # epochs: 300 
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if 'OrderedDict' == checkpoint.__class__.__name__:
                for k in list(checkpoint.keys()):
                    checkpoint['module.' + k] = checkpoint[k]
                    del checkpoint[k]
                checkpoint = {'state_dict': checkpoint}

            args.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            best_acc1 = checkpoint['best_acc1'] if 'best_acc1' in checkpoint else 0
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'train' if args.valset else 'val')
    valdir = valdir if args.valdir is None else args.valdir
    if args.dataset == 'inat18': traindir = valdir = args.data
    augtype = 'lt' if args.ltaug else None
    train_transform, val_transform = get_transforms(args.dataset, augtype)
    _, train_sampler, train_loader = get_dataloader(args, traindir, train_transform)
    _, _, val_loader = get_dataloader(args, valdir, val_transform, is_val=True)

    if args.evaluate and args.taunorm:
        tau_validate(val_loader, model, args, train_loader.dataset.get_cls_cnts())
        return

    if args.evaluate:
        top1, percls_acc = validate(val_loader, model, criterion, args)
        shot_acc = util.shot_acc(percls_acc, train_loader.dataset.get_cls_cnts())
        shot_acc = [acc*100 for acc in shot_acc]
        print(' \t '.join(['{:.3f}'] * 4).format(*shot_acc, top1))
        if args.save_acc:
            f = args.resume.replace('clsmodel', 'accs')
            f = f.replace('featmodel', 'accs')
            f = f.replace('.pth.tar', '.pk')
            f = f.replace('.pth', '.pk')
            util.save_obj(f, {'acc': percls_acc,
                              'cnt': train_loader.dataset.get_cls_cnts()})
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=args.name)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    if args.eval_mode:
        model.eval()
    else:
        # switch to train mode
        model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        topk = (1,2) if args.rotation else (1, 5)
        acc1, acc5 = accuracy(output, target, topk=topk)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    percls_accs = util.AverageMeters(args.num_classes)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            topk = (1,2) if args.rotation else (1, 5)
            (acc1, acc5), tgt, correct = \
                accuracy(output, target, topk=topk, detail=True)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            percls_accs.update(tgt, correct)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if args.evaluate:
        return top1.avg, percls_accs.get_avgs()
    else:
        return top1.avg


def tau_validate(val_loader, model, args, cls_cnts):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='TauTest: ')

    # Change classifier
    print('=> replacing classifier')
    if 'DataParallel' in model.__class__.__name__:
        model.module.fc = util.TauLayer(model.module.fc)
        if len(model.device_ids) == 1:
            model.module.to(model.src_device_obj)
        model.cuda()
        tau = model.module.fc.tau
    else:
        model.fc = util.TauLayer(model.fc)
        model.cuda()
        tau = model.fc.tau
    
    # track percls accuracies 
    percls_accs = [util.AverageMeters(args.num_classes)\
                   for _ in range(len(tau))]
    top1_accs = [AverageMeter('Acc@1', ':6.2f') for _ in range(len(tau))]
    top5_accs = [AverageMeter('Acc@5', ':6.2f') for _ in range(len(tau))]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            # split the output based on number of taus
            tau_outputs = torch.chunk(output, len(tau), dim=1)
            for ti, tau_out in enumerate(tau_outputs):
                # measure accuracy
                (acc1, acc5), tgt, correct = accuracy(
                    tau_out, target, topk=(1, 5), detail=True)
                percls_accs[ti].update(tgt, correct)
                top1_accs[ti].update(acc1[0], images.size(0))
                top5_accs[ti].update(acc5[0], images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    for tau, percls_acc, top1, top5 in \
        zip(tau, percls_accs, top1_accs, top5_accs):
        shot_acc = util.shot_acc(percls_acc.get_avgs(), cls_cnts)
        shot_acc = [acc*100 for acc in shot_acc]
        print(' \t '.join(['{:.3f}'] * 5).format(*shot_acc, top1.avg, top5.avg))

if __name__ == '__main__':
    main(main_worker, parser)
