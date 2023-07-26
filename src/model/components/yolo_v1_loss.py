import torch
import torch.nn as nn
from utils.utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S ## Grid size
        self.B = B ## Number of bounding boxes per grid cell
        self.C = C ## Number of classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        ious = []
        for i in range(self.B):
            iou = intersection_over_union(predictions[..., self.C + i * 5 + 1: self.C + i * 5 + 1 + 4], 
                                          target[..., 21:25], box_format="midpoint")
            ious.append(iou.unseqeeze(0)) ## (1, Batch, S, S, 1)
        ious = torch.cat(ious, dim=0)  ## (N, Batch, S, S, 1)

        # Find the best box
        iou_maxes, best_box = torch.max(ious, dim=0) ## (Batch, S, S, 1),
        exists_box = target[..., 20].unsqueeze(3) ## (Batch, S, S, 1) 

        #########################
        # BOX LOCALIZATION LOSS #
        #########################
        box_predictions = torch.zeros(best_box.shape[0], best_box.shape[1], best_box.shape[2], 4) ##(Batch, S, S, 4)
        
        BATCH = predictions.shape[0]
        for i in range(BATCH):
            for j in range(self.S):
                for k in range(self.S):
                    box_predictions[i, j, k, :] = predictions[BATCH, i, j, k, self.C + best_box[BATCH, i, j, k] * 5 + 1:
                                                  self.C + best_box[BATCH, i, j, k] * 5 + 1 + 4]
        
        box_predictions = exists_box * box_predictions ## (Batch, S, S, 1) * (Batch, S, S, 4)

        ## Take the square root of the width and height, we need to use abs because  width and height can be negative
        box_predictions[..., 2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]) + 1e6)

        box_targets = exists_box * target[..., 21:25] ## (Batch, S, S, 1) * (Batch, S, S, 4)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4) ## Actually, we dont need to do this but just do it for our logic
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2), # (N, S, S, 4) -> (N*S*S, 4)
            torch.flatten(box_targets, end_dim = -2), # (N, S, S, 4) -> (N*S*S, 4)
        )

        #########################
        #    FOR OBJECT LOSS    #
        #########################
        pred_box = torch.zeros(best_box.shape) ## (Batch, S, S, 1)
        for i in range(BATCH):
            for j in range(self.S):
                for k in range(self.S):
                    pred_box[i, j, k, :] = predictions[BATCH, i, j, k, self.C]

        pred_box = exists_box * pred_box ## (Batch, S, S, 1) * (Batch, S, S, 1)

        object_loss = self.mse(
            torch.flatten(exists_box, pred_box),
            torch.flatten(exists_box, target[..., self.C:self.C + 1]),
        )

        #########################
        #  FOR NO OBJECT LOSS   #
        #########################
        no_object_loss = 0
        # self.mse (
        #     torch.flatten((1 - exists_box) * predictions[..., self.C:self.C + 1]),
        #     torch.flatten((1 - exists_box) * target[..., self.C:self.C + 1])  ## (Batch, S, S, 1) * (Batch, S, S, 1)
        # )

        for i in range(self.B):
            no_object_loss += self.mse(
                torch.flatten((1 - exists_box) * predictions[..., self.C + 5 * i : self.C + 5 * i + 1]),
                torch.flatten((1 - exists_box) * target[..., self.C : self.C + 1])
            )

        #########################
        #    FOR CLASS LOSS    #
        #########################

        class_loss = self.mse(
            torch.flatten(predictions[..., :self.C], end_dim = -2), # (N, S, S, Class) -> (N*S*S, Class)
            torch.flatten(target[..., :self.C], end_dim = -2), # (N, S, S, Class) -> (N*S*S, Class)
        )

        loss = (
            self.lambda_coord * box_loss # First two rows of the paper
            + object_loss # Third rows of the paper
            + self.lambda_noobj * no_object_loss # Fourth rows of the paper
            + class_loss # Fifth rows of the paper
        )

        return loss
