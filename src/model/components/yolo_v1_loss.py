import torch
import torch.nn as nn
from yolo_v1 import YoloV1
from src.utils.utils import intersection_over_union

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
                                          target[..., self.C + 1: self.C + 1 + 4], box_format="midpoint")
            ious.append(iou.unsqueeze(0)) ## (1, Batch, S, S, 1)
            # print(iou.shape)
        ious = torch.cat(ious, dim=0)  ## (N, Batch, S, S, 1)

        # Find the best box
        iou_maxes, best_box = torch.max(ious, dim=0) ## (Batch, S, S, 1),
        exists_box = target[..., self.C].unsqueeze(3) ## (Batch, S, S, 1) 
        # print(ious.shape)

        #########################
        # BOX LOCALIZATION LOSS #
        #########################
        box_predictions = torch.zeros(best_box.shape[0], best_box.shape[1], best_box.shape[2], 4) ##(Batch, S, S, 4)
        
        BATCH = predictions.shape[0]
        for i in range(BATCH):
            for j in range(self.S):
                for k in range(self.S):
                    id = int(best_box[i, j, k].item())
                    box_predictions[i, j, k, :] = predictions[i, j, k, self.C + id * 5 + 1:
                                                  self.C + id * 5 + 1 + 4].clone()
        
        box_predictions = exists_box * box_predictions ## (Batch, S, S, 1) * (Batch, S, S, 4)

        ## Take the square root of the width and height, we need to use abs because  width and height can be negative
        box_predictions[..., 2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]) + 1e6)

        box_targets = exists_box * target[..., self.C + 1 : self.C + 1 + 4] ## (Batch, S, S, 1) * (Batch, S, S, 4)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4) ## Actually, we dont need to do this but just do it for our logic
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2), # (N, S, S, 4) -> (N*S*S, 4)
            torch.flatten(box_targets, end_dim = -2), # (N, S, S, 4) -> (N*S*S, 4)
        )

        # print(box_loss)
        #########################
        #    FOR OBJECT LOSS    #
        #########################
        pred_box = torch.zeros(best_box.shape) ## (Batch, S, S, 1)
        for i in range(BATCH):
            for j in range(self.S):
                for k in range(self.S):
                    id = int(best_box[i, j, k].item())
                    pred_box[i, j, k, :] = predictions[i, j, k, self.C + id * 5].clone()

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box), ## (Batch, S, S, 1) * (Batch, S, S, 1)
            torch.flatten(exists_box * target[..., self.C:self.C + 1]),
        )

        # print(object_loss)
        #########################
        #  FOR NO OBJECT LOSS   #
        #########################

        ## We need to sum all of predicted boxes in a grid cell because we want all of object confidence of predicted boxes to be zero!
        no_object_loss = 0
        for i in range(self.B):
            no_object_loss += self.mse(
                torch.flatten((1 - exists_box) * predictions[..., self.C + 5 * i : self.C + 5 * i + 1]),
                torch.flatten((1 - exists_box) * target[..., self.C : self.C + 1])
            )

        # no_object_loss = self.mse(
        #     torch.flatten((1 - exists_box) * pred_box), ## (Batch, S, S, 1) * (Batch, S, S, 1)
        #     torch.flatten((1 - exists_box) * target[..., self.C : self.C + 1])
        # )

        # print(no_object_loss)
        #########################
        #    FOR CLASS LOSS    #
        #########################

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim = -2), # (N, S, S, Class) -> (N*S*S, Class)
            torch.flatten(exists_box * target[..., :self.C], end_dim = -2), # (N, S, S, Class) -> (N*S*S, Class)
        )

        # print(class_loss)
        #########################
        #      FINAL LOSS       #
        #########################
        loss = (
            self.lambda_coord * box_loss # First two rows of the paper
            + object_loss # Third rows of the paper
            + self.lambda_noobj * no_object_loss # Fourth rows of the paper
            + class_loss # Fifth rows of the paper
        )

        return loss
    
def main():
    S = 10
    B = 10
    C = 10
    loss_fn = YoloLoss(S = S, B = B, C = C)

    predictions = torch.randn((2, S, S, C + B * 5)) ## (Batch, S, S, C + B * 5)
    target = torch.abs(torch.randn((2, S, S, C + 5))) ## (Batch, S, S, C, B)

    res = loss_fn(predictions, target)
    print(res)

if __name__ == "__main__":
    main()
