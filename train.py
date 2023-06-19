import os
import pdb
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim
from tqdm import tqdm

import dataloaders
from utils.metrics import Evaluator
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args


def main():
    # get dataloader
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_autodeeplab_args()
    model_fname = f'{args.checkname}_epoch%d.pth'
    if args.checkname is None:
        model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(
            args.backbone, args.dataset, args.exp)
    if args.dataset == 'solis':
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        dataset_loader, val_loader = dataloaders.make_data_loader(
            args, **kwargs)
        args.num_classes = 2
    elif args.dataset == 'pascal':
        raise NotImplementedError
    elif args.dataset == 'cityscapes':
        kwargs = {'num_workers': args.workers,
                  'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(
            args, **kwargs)
        args.num_classes = num_classes
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    saver = Saver(args)
    saver.save_experiment_config()
    # Define Tensorboard Summary
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()
    evaluator = Evaluator(2)

    # get model
    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(
            args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu) *
                         args.crop_size[0] * args.crop_size[1]) // 16)
    criterion = build_criterion(args)

    # model = nn.DataParallel(model).cuda()
    model.cuda()
    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    # optimizer = optim.SGD(model.module.parameters(),
    #                       lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iteration = len(dataset_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(dataset_loader))
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                '=> no checkpoint found at {0}'.format(args.resume))

    # train

    best_pred = 0.0

    print('Starting Epoch:', start_epoch)

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        model.train()
        evaluator.reset()
        tbar = tqdm(dataset_loader)
        num_img_tr = len(dataset_loader)
        for i, sample in enumerate(tbar):
            cur_iter = epoch * len(tbar) + i
            scheduler(optimizer, cur_iter)
            if args.dataset == 'solis':
                inputs, target = sample
            else:
                inputs, target = sample['image'], sample['label']
            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # outputs = outputs.data.cpu().numpy()
            # target = target.cpu().numpy()
            # outputs = np.argmax(outputs, axis=1)
            evaluator.add_batch(target.cpu().numpy(), np.argmax(
                outputs.data.cpu().numpy(), axis=1))

            tbar.set_description('Train loss: %.3f' % (losses.sum / (i + 1)))

            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                summary.visualize_image(
                    writer, args.dataset, inputs, target, outputs, global_step)

        print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
            epoch + 1, i + 1, len(dataset_loader), scheduler.get_lr(optimizer), loss=losses))

        train_mIou = write_epoch(writer, epoch, losses.avg, 'train')

        # Validate after every 5 epochs
        if epoch % 5 == 0:
            val_losses = AverageMeter()
            model.eval()  # set model to evaluation mode
            evaluator.reset()
            with torch.no_grad():  # disable gradient computation
                for i, sample in enumerate(val_loader):
                    if args.dataset == 'solis':
                        inputs, target = sample
                    else:
                        inputs, target = sample['image'], sample['label']
                    if args.cuda:
                        inputs, target = inputs.cuda(), target.cuda()
                    outputs = model(inputs)
                    val_loss = criterion(outputs, target)
                    val_losses.update(val_loss.item(), args.batch_size)

                    # outputs = outputs.data.cpu().numpy()
                    # target = target.cpu().numpy()
                    # outputs = np.argmax(outputs, axis=1)
                    # evaluator.add_batch(target, outputs)
                    evaluator.add_batch(target.cpu().numpy(), np.argmax(
                        outputs.data.cpu().numpy(), axis=1))

            print('Validation: epoch: {0}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                epoch + 1, loss=val_losses))
            is_best = False
            val_mIoU = write_epoch(writer, epoch, val_losses.avg, 'val')
            if val_mIoU > best_pred:
                is_best = True
                best_pred = val_mIoU
        # save model

        if epoch > args.epochs - 50 or epoch % 5 == 0:
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, is_best, filename=f'checkpoint_{epoch + 1}.pth.tar')
            # torch.save({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # }, model_fname % (epoch + 1))
        print('reset local total loss!')


def write_epoch(writer, evaluator, name, epoch, loss, cur_iter):

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    F1 = evaluator.F1_score()
    writer.add_scalar(f'{name}/total_loss_epoch', loss, epoch)
    writer.add_scalar(f'{name}/mIoU', mIoU, epoch)
    writer.add_scalar(f'{name}/Acc', Acc, epoch)
    writer.add_scalar(f'{name}/Acc_class', Acc_class, epoch)
    writer.add_scalar(f'{name}/fwIoU', FWIoU, epoch)
    writer.add_scalar(f'{name}/F1', F1, epoch)
    return mIoU


if __name__ == "__main__":
    main()
