import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.dataloader_utils import decode_seg_map_sequence
import torch.distributed as dist


class TensorboardSummary(object):
    def __init__(self, directory, use_dist=False):
        self.directory = directory
        self.use_dist = use_dist

    def create_summary(self):
        writer = SummaryWriter(logdir=os.path.join(self.directory))
        return writer

    # def visualize_image(self, writer, dataset, image, target, output, global_step):
    #     if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
    #         # TODO: this might be overkill
    #         if image.shape[1] == 12:
    #             image2 = image[:, :3, :, :].clone()
    #             grid_image = make_grid(
    #                 image2[:3].clone().cpu().data, 3, normalize=True)
    #         else:
    #             grid_image = make_grid(
    #                 image[:3].clone().cpu().data, 3, normalize=True)
    #         writer.add_image('Image', grid_image, global_step)
    #         grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
    #                                                        dataset=dataset), 3, normalize=False, range=(0, 255))
    #         # grid_image = make_grid(torch.max(output[:3], 1)[
    #         #                        1].detach().cpu().numpy())
    #         writer.add_image('Predicted label', grid_image, global_step)
    #         grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
    #                                                        dataset=dataset), 3, normalize=False, range=(0, 255))
    #         # grid_image = make_grid(torch.squeeze(
    #         #     target[:3], 1).detach().cpu().numpy())

    #         writer.add_image('Groundtruth label', grid_image, global_step)
    def visualize_image(self, writer, dataset, image, target, output, global_step):
        if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
            # Extract RGB bands and stack them in the correct order
            if image.shape[1] == 12:
                red = image[:, 3:4, :, :]
                green = image[:, 2:3, :, :]
                blue = image[:, 1:2, :, :]
                rgb_image = torch.cat((red, green, blue), 1)
                grid_image = make_grid(
                    rgb_image[:3].clone().cpu().data, 3, normalize=True)
            else:
                grid_image = make_grid(
                    image[:3].clone().cpu().data, 3, normalize=True)

            writer.add_image('Image', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Predicted label', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Groundtruth label', grid_image, global_step)
