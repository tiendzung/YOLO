import torch
import torch.nn as nn
from utils.utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        best_box = 0
        ious = []
        for i in range(self.B):
            iou = intersection_over_union(predictions[..., self.C + i * 5 + 1: self.C + i * 5 + 1 + 4], 
                                          target[..., 21:25], box_format="midpoint").unsqueeze(0)
            ious.append(iou)
        ios = torch.cat(ious, dim=0)

        # Find the best box
        ious_maxes, best_box = torch.max(ious, dim=0)
