import torch
import torch.nn as nn
from tqdm.notebook import tqdm as tqdm
from sklearn.metrics import accuracy_score
import time


class IOU(nn.Module):
    '''calculates intersection over union for a batch of images'''

    def __init__(self):
        super().__init__()

    def forward(self, pr: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
        # tr - coordinates of a true box, pr - coordinates of a predicted box

        # a tensor with the areas of target bounding boxes for each image:
        true_area = (tr[:, 2] - tr[:, 0]) * (
                tr[:, 3] - tr[:, 1])
        # a tensor  with the areas of predicted bounding boxes
        pred_area = (pr[:, 2] - pr[:, 0]) * (
                pr[:, 3] - pr[:, 1])

        # xi, yi - coordinates of the intersection of true and predicted bounding boxes
        xi_min, yi_min = torch.max(tr[:, 0], pr[:, 0]), torch.max(tr[:, 1], pr[:, 1])
        xi_max, yi_max = torch.min(tr[:, 2], pr[:, 2]), torch.min(tr[:, 3], pr[:, 3])
        width = xi_max - xi_min
        height = yi_max - yi_min
        # case if there is no intersection:
        width[width < 0] = 0
        height[height < 0] = 0

        # a tensor with the intersection area for each image in a batch:
        intersection = width * height
        # a tensor with the union area for each image in a batch:
        union = true_area + pred_area - intersection
        # a tensor with the mean intersection for a batch
        iou = (intersection / union).mean()
        return iou


def training(net, n_epoch, lr, dataloader, device, transfer_learning=None):
    # classification loss:
    loss_f = torch.nn.BCELoss()
    # detection loss:
    iou = IOU()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net = net.to(device)

    for epoch in tqdm(range(n_epoch)):
        loader = iter(dataloader)
        total_loss = 0

        for idx, (image, target) in tqdm(enumerate(loader)):
            image = image.to(device)
            target['label'] = target['label'].to(device)
            target['boxes'] = target['boxes'].to(device)
            optimizer.zero_grad()
            if transfer_learning:
                # it will be used inception_v3 for transfer learning.
                # Because of an auxiliary classifier we need to unpack three values:
                (pr_class, pr_box), _ = net(image)
                pr_class = pr_class.squeeze()
                pr_box = pr_box.squeeze()
            else:
                pr_class, pr_box = net(image)
                pr_class = pr_class.squeeze()
                pr_box = pr_box.squeeze()
            # classification loss:
            loss_cl = loss_f(pr_class, target['label'])
            # detection loss:
            loss_det = -iou(pr_box, target['boxes'])
            # total_loss:
            loss = loss_cl + loss_det

            loss.backward()
            total_loss += loss.item()

            optimizer.step()

        if epoch % 5 == 0:
            print('epoch: ', epoch + 1, '\n', 'Losses for last batch:',
                  '\n', 'classification loss:', loss_cl, '\n', 'total loss:', total_loss)
    return net


def validation(net, dataloader, train_loader, device, threshold=0.51):
    t0 = time.time()
    test_loader = iter(dataloader)

    iou = IOU()
    labels = []
    pred_labels = []
    total_miou = 0

    net.eval()
    with torch.no_grad():
        for idx, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target['label'] = target['label'].to(device)
            target['boxes'] = target['boxes'].to(device)

            pr_class, pr_box = net(image)
            pr_class = (pr_class > threshold).float()

            labels.extend(target['label'].tolist())
            pred_labels.extend(pr_class.squeeze().tolist())
            io = iou(pr_box, target['boxes'])
            total_miou += io

    accuracy = accuracy_score(labels, pred_labels)
    t = time.time() - t0
    print('mIoU %0.0f%%,' % ((total_miou / (idx + 1)) * 100),
          'classification accuracy %0.0f%%,' % (accuracy * 100),
          '%0.2fs,' % (t),
          '%0.0f train,' % (len(train_loader.dataset)),
          '%0.0f valid.' % (len(dataloader.dataset)))
