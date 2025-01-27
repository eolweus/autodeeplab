import os
import shutil
import torch
from collections import OrderedDict
import glob
import torch.distributed as dist


class Saver(object):

    def __init__(self, args, use_dist=False):
        self.args = args
        self.use_dist = use_dist
        self.directory = os.path.join('../run', args.dataset, args.checkname)
        self.runs = sorted(
            glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = max([int(x.split('_')[-1])
                     for x in self.runs]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(
            self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
            print("saving checkpoint:", filename)
            torch.save(state, filename)
            if is_best:
                best_pred = state['best_pred']
                best_state = f'best_pred: {best_pred}, epoch: {state["epoch"]}'
                with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                    f.write(best_state)
                if self.runs:
                    previous_miou = [0.0]
                    for run in self.runs:
                        run_id = run.split('_')[-1]
                        path = os.path.join(self.directory, 'experiment_{}'.format(
                            str(run_id)), 'best_pred.txt')
                        if os.path.exists(path):
                            with open(path, 'r') as f:
                                text = f.readline()
                                try:
                                    miou = float(text)
                                except:
                                    miou = float(
                                        next(s.split(": ")[1] for s in text.split(", ") if "best_pred" in s))

                                previous_miou.append(miou)
                        else:
                            continue
                    max_miou = max(previous_miou)
                    if best_pred > max_miou:
                        shutil.copyfile(filename, os.path.join(
                            self.directory, 'model_best.pth.tar'))
                else:
                    shutil.copyfile(filename, os.path.join(
                        self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
            if self.args.autodeeplab == 'search':
                logfile = os.path.join(self.experiment_dir, 'parameters.txt')
                log_file = open(logfile, 'w')
                p = OrderedDict()
                p['datset'] = self.args.dataset
                p['backbone'] = self.args.backbone
                p['out_stride'] = self.args.out_stride
                p['lr'] = self.args.lr
                p['lr_scheduler'] = self.args.lr_scheduler
                p['loss_type'] = self.args.loss_type
                p['epochs'] = self.args.epochs
                p['alpha_epoch'] = self.args.alpha_epoch
                p['batch_size'] = self.args.batch_size
                p['workers'] = self.args.workers
                p['resize'] = self.args.resize
                p['crop_size'] = self.args.crop_size
                p['num_images'] = self.args.num_images
                p['subset_ratio'] = self.args.subset_ratio
                p['num_bands'] = self.args.num_bands
                for key, val in p.items():
                    log_file.write(key + ':' + str(val) + '\n')
                log_file.close()
            else:
                logfile = os.path.join(self.experiment_dir, 'parameters.txt')
                log_file = open(logfile, 'w')
                p = OrderedDict()
                p['datset'] = self.args.dataset
                p['backbone'] = self.args.backbone
                p['base_lr'] = self.args.base_lr
                p['epochs'] = self.args.epochs
                p['batch_size'] = self.args.batch_size
                p['workers'] = self.args.workers
                p['resize'] = self.args.resize
                p['crop_size'] = self.args.crop_size
                p['num_images'] = self.args.num_images
                p['subset_ratio'] = self.args.subset_ratio
                p['num_bands'] = self.args.num_bands
                for key, val in p.items():
                    log_file.write(key + ':' + str(val) + '\n')
                log_file.close()
