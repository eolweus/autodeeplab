import os
import pdb
import random
import warnings
import numpy as np

from PIL import Image
import rasterio
from rasterio.transform import from_origin

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders.dataloader_utils import decode_seg_map_sequence, decode_segmap

from modeling.backbone.solis_models import deeplabv3_resnet50

import dataloaders
from utils.jp2_to_gtiff import batch_convert
from utils.metrics import Evaluator
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from retrain_model.build_autodeeplab import Retrain_Autodeeplab, Retrain_Autodeeplab2
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args


def main():
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_autodeeplab_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # get dataloader
    if args.dataset == 'solis':
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        dataset_loader = dataloaders.make_data_loader(
            args, **kwargs)
        args.num_classes = 2
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # evaluator = Evaluator(2)

    # get model
    if args.backbone == 'autodeeplab':
        if args.use_new_model:
            model = Retrain_Autodeeplab2(args)
        else:
            model = Retrain_Autodeeplab(args)

    elif args.backbone == 'solis':
        model = deeplabv3_resnet50()
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    criterion = build_criterion(args)

    model.cuda()
    model.eval()

    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            # start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                '=> no checkpoint found at {0}'.format(args.resume))

    print('start inference')
    with torch.no_grad():  # disable gradient computation
        for i, sample in enumerate(dataset_loader):
            inputs, target, name = sample
            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()
            outputs = model(inputs)
            if args.backbone == 'solis':
                outputs = outputs['out']

            outputs.data.cpu().numpy()

            # Iterate over each image in the batch
            output_masks = decode_seg_map_sequence(torch.max(outputs, 1)[1].detach().cpu().numpy(),
                                                   dataset="solis")

            for j in range(outputs.shape[0]):

                image = output_masks[j] * 255
                # output_masks_normalized = (
                #     output_masks[j] - output_masks[j].min()) / (output_masks[j].max() - output_masks[j].min())
                # image = output_masks_normalized * 255

                print(output_masks[j].shape)
                output_mask_image = Image.fromarray(
                    image.numpy().transpose(
                        (1, 2, 0)).astype(np.uint8)
                )
                n = name[j].split('.')[0]
                # Save the output mask as a PNG file
                output_mask_path = os.path.abspath(
                    f"/cluster/home/erlingfo/autodeeplab/test_images/pred/{n}.png"
                )
                output_mask_image.save(output_mask_path)

                # output_mask = outputs[j].data.cpu().numpy().astype(np.uint8)
                output_mask = torch.max(outputs, 1)[
                    1].detach().cpu().numpy()[j]

                height, width = output_mask.shape
                # This sets the geotransformation. Adjust as needed.
                transform = from_origin(0, 0, 1, 1)

                # Define the properties of the .tif file
                kwargs = {
                    'driver': 'GTiff',  # GeoTiff driver
                    'height': height,
                    'width': width,
                    # number of bands (1 because your mask is grayscale)
                    'count': 1,
                    'dtype': output_mask.dtype,  # datatype, should match the numpy array's dtype
                    # Coordinate reference system. Adjust as needed.
                    'crs': '+proj=latlong',
                    'transform': transform,
                }

                output_tif_path = f"/cluster/home/erlingfo/autodeeplab/test_images/tif_pred/{n}.tif"

                with rasterio.open(output_tif_path, 'w', **kwargs) as dst:
                    # Writing the mask to band 1 of the GeoTiff file
                    dst.write(output_mask, 1)

        # walk through the target folder and check if the target image is there as a tif file and if not convert it
        # to a tif file
        target_path = os.path.abspath(
            f"/cluster/home/erlingfo/autodeeplab/test_images/target"
        )
        if not os.path.exists(target_path + f'/{n}.tif'):
            batch_convert(target_path)
    print('inference done')


if __name__ == '__main__':
    main()
