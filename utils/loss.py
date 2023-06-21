import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.cuda = cuda
        self.batch_average = batch_average
        self.criterion = None

    def build_loss(self, mode='ce'):
        """Choices: ['ce', 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        # n, c, h, w = logit.size()

        if self.criterion is None:
            reduction = 'mean' if self.size_average else 'none'
            self.criterion = nn.CrossEntropyLoss(
                weight=self.weight, ignore_index=self.ignore_index, reduction=reduction)
            if self.cuda:
                self.criterion = self.criterion.cuda()

        loss = self.criterion(logit, target.long())

        if self.batch_average and not self.size_average:
            loss = loss.mean()

        return loss

    # def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
    #     n, c, h, w = logit.size()
    #     criterion = nn.CrossEntropyLoss(
    #         weight=self.weight, ignore_index=self.ignore_index, reduction='none')

    #     if self.cuda:
    #         criterion = criterion.cuda()

    #     logpt = -criterion(logit, target.long())
    #     pt = torch.exp(logpt)

    #     # Create mask for positive samples (ignore_index is assumed to be negative)
    #     mask = (target != self.ignore_index).float()
    #     alpha_t = torch.ones_like(target) * alpha
    #     alpha_t = alpha_t * mask

    #     loss = -alpha_t * ((1 - pt) ** gamma) * logpt
    #     # Apply mask to exclude the ignore_index samples from the loss
    #     loss = loss * mask

    #     if self.size_average:
    #         loss = loss.sum() / mask.sum()  # Apply mean reduction based on the mask

    #     if self.batch_average:
    #         loss /= n

    #     return loss

    # def CrossEntropyLoss(self, logit, target):
    #     n, c, h, w = logit.size()
    #     criterion = nn.CrossEntropyLoss(
    #         weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
    #     if self.cuda:
    #         criterion = criterion.cuda()

    #     loss = criterion(logit, target.long())

    #     return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25, reduction='mean'):
        if self.cuda:
            logit = logit.cuda()
            target = target.cuda()

        n, c, h, w = logit.size()
        target_one_hot = torch.zeros(n, c, h, w, device=logit.device)
        target_one_hot.scatter_(1, target.unsqueeze(1).long(), 1)

        loss = sigmoid_focal_loss(
            logit, target_one_hot, alpha, gamma, reduction)

        return loss

    # def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
    #     n, c, h, w = logit.size()
    #     criterion = nn.CrossEntropyLoss(
    #         weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
    #     if self.cuda:
    #         criterion = criterion.cuda()

    #     logpt = -criterion(logit, target.long())
    #     pt = torch.exp(logpt)
    #     if alpha is not None:
    #         logpt *= alpha
    #     loss = -((1 - pt) ** gamma) * logpt

    #     if self.batch_average:
    #         loss /= n

    #     return loss


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_index=255, cuda=True, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if cuda:
            self.criteria = self.criteria.cuda()

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.clone()
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels.clone()
            invalid_mask = labels_cpu == self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu.long()]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min] < self.thresh else sorteds[self.n_min]
            labels[picks > thresh] = self.ignore_lb
        # labels = labels.clone()
        loss = self.criteria(logits, labels.long())
        return loss


def build_criterion(args):
    print("=> Trying bulid {:}loss".format(args.criterion))
    if args.criterion == 'Ohem':
        return OhemCELoss(thresh=args.thresh, n_min=args.n_min, cuda=True)
    elif args.criterion == 'ce':
        return SegmentationLosses(weight=None, cuda=True).build_loss('ce')
    else:
        raise ValueError('unknown criterion : {:}'.format(args.criterion))


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
