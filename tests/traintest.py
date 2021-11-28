import pytest
from train_fuctions import IOU
import numpy as np
import torch


def test_IOU():
    tensor1 = torch.Tensor([[100, 200, 300, 400],
                            [10, 20, 30, 40],
                            [95, 180, 320, 410]])
    tensor2 = torch.Tensor([[95, 180, 320, 410],
                            [105, 200, 300, 420],
                            [195, 170, 310, 460]])
    ans1 = torch.Tensor([1])
    iou = IOU()
    assert iou.forward(tensor1, tensor1) == ans1
    assert iou.forward(tensor1[0].unsqueeze(dim=0), tensor2[1].unsqueeze(dim=0)) == pytest.approx(0.85, 0.1)
    assert iou.forward(tensor1[0].unsqueeze(dim=0), tensor2[2].unsqueeze(dim=0)) < ans1
    assert iou.forward(tensor1[1].unsqueeze(dim=0), tensor2[2].unsqueeze(dim=0)) < ans1
    assert iou.forward(tensor1[2].unsqueeze(dim=0), tensor2[1].unsqueeze(dim=0)) == pytest.approx(0.75, 0.1)
    assert iou.forward(tensor1[1].unsqueeze(dim=0), tensor2[0].unsqueeze(dim=0)) == torch.Tensor([0])
    assert iou.forward(tensor1[2].unsqueeze(dim=0), tensor2[0].unsqueeze(dim=0)) == ans1

test_IOU()


