# from dataloaders.datasets import cityscapes, kd, coco, combine_dbs, pascal, sbd, solis
from dataloaders.datasets import cityscapes, combine_dbs, pascal, sbd, solis_data_module
from torch.utils.data import DataLoader
import torch.utils.data.distributed


def make_data_loader(args, **kwargs):
    if args.dist:
        print("=> Using Distribued Sampler")
        if args.dataset == 'cityscapes':
            if args.autodeeplab == 'search':
                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
                num_class = train_set1.NUM_CLASSES
                sampler1 = torch.utils.data.distributed.DistributedSampler(
                    train_set1)
                sampler2 = torch.utils.data.distributed.DistributedSampler(
                    train_set2)
                train_loader1 = DataLoader(
                    train_set1, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)
                train_loader2 = DataLoader(
                    train_set2, batch_size=args.batch_size, shuffle=False, sampler=sampler2, **kwargs)

            elif args.autodeeplab == 'train':
                train_set = cityscapes.CityscapesSegmentation(
                    args, split='retrain')
                num_class = train_set.NUM_CLASSES
                sampler1 = torch.utils.data.distributed.DistributedSampler(
                    train_set)
                train_loader = DataLoader(
                    train_set, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)

            else:
                raise Exception('autodeeplab param not set properly')

            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')
            sampler3 = torch.utils.data.distributed.DistributedSampler(val_set)
            sampler4 = torch.utils.data.distributed.DistributedSampler(
                test_set)
            val_loader = DataLoader(
                val_set, batch_size=args.batch_size, shuffle=False, sampler=sampler3, **kwargs)
            test_loader = DataLoader(
                test_set, batch_size=args.batch_size, shuffle=False, sampler=sampler4, **kwargs)

            if args.autodeeplab == 'search':
                return train_loader1, train_loader2, val_loader, test_loader, num_class
            elif args.autodeeplab == 'train':
                return train_loader, num_class, sampler1
        else:
            raise NotImplementedError

    else:
        if args.dataset == 'pascal':
            train_set = pascal.VOCSegmentation(args, split='train')
            val_set = pascal.VOCSegmentation(args, split='val')
            if args.use_sbd:
                sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
                train_set = combine_dbs.CombineDBs(
                    [train_set, sbd_train], excluded=[val_set])

            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(
                val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = None

            return train_loader, train_loader, val_loader, test_loader, num_class

        elif args.dataset == 'cityscapes':
            if args.autodeeplab == 'search':
                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
                num_class = train_set1.NUM_CLASSES
                train_loader1 = DataLoader(
                    train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
                train_loader2 = DataLoader(
                    train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
            elif args.autodeeplab == 'train':
                train_set = cityscapes.CityscapesSegmentation(
                    args, split='retrain')
                num_class = train_set.NUM_CLASSES
                train_loader = DataLoader(
                    train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            else:
                raise Exception('autodeeplab param not set properly')

            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')
            val_loader = DataLoader(
                val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(
                test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            if args.autodeeplab == 'search':
                return train_loader1, train_loader2, val_loader, test_loader, num_class
            elif args.autodeeplab == 'train':
                return train_loader, num_class

        elif args.dataset == 'solis':
            dataset = solis_data_module.ChipFolderSegmentationDatamodule(args)
            if args.autodeeplab == 'infer':
                data_loader = dataset.predict_dataloader()
                return data_loader

            elif args.use_ab and args.autodeeplab == 'search':
                train_loader1, train_loader2 = dataset.train_dataloader_ab()
            elif args.autodeeplab == 'train':
                train_loader1 = dataset.train_dataloader()
            else:
                train_loader1 = dataset.train_dataloader()
                train_loader2 = train_loader1
            val_loader = dataset.val_dataloader()
            num_class = 2
            test_loader = None
            if args.autodeeplab == 'train':
                return train_loader1, val_loader
            return train_loader1, train_loader2, val_loader, test_loader, num_class

        else:
            raise NotImplementedError
