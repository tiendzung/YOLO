import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    """
    ## box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    if box_format == "midpoints":
        box1_x1 = boxes_preds[...,0] - boxes_preds[...,2] / 2
        box1_y1 = boxes_preds[...,1] - boxes_preds[...,3] / 2
        box1_x2 = boxes_preds[...,0] + boxes_preds[...,2] / 2
        box1_y2 = boxes_preds[...,1] + boxes_preds[...,3] / 2

        box2_x1 = boxes_labels[...,0] - boxes_labels[...,2] / 2
        box2_y1 = boxes_labels[...,1] - boxes_labels[...,3] / 2
        box2_x2 = boxes_labels[...,0] + boxes_labels[...,2] / 2
        box2_y2 = boxes_labels[...,1] + boxes_labels[...,3] / 2
        
    if box_format == "corners":
        box1_x1 = boxes_preds[...,0]
        box1_y1 = boxes_preds[...,1]
        box1_x2 = boxes_preds[...,2]
        box1_y2 = boxes_preds[...,3]

        box2_x1 = boxes_labels[...,0]
        box2_y1 = boxes_labels[...,1]
        box2_x2 = boxes_labels[...,2]
        box2_y2 = boxes_labels[...,3]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    ## .clamp(0) is for the case when there is no intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
