import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=3, weight_map=None, eps=1e-8):
        dice_loss = soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map)
        return dice_loss


def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
    # dice_loss = 1.0 - torch.mean(dice_score)
    # return dice_loss
    dice_score = torch.mean(-torch.log(dice_score))
    return dice_score


def val_dice_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_mean_score = torch.mean(dice_score)

    return dice_mean_score

def val_jaccard_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    jaccard_score = intersect / (ref_vol + seg_vol + 1.0) # remove 2* from denominator 
    # Dice = 2 |A∩B| / (|A|+|B|) = 2 TP / (2 TP + FP + FN)
    # Jaccard = |A∩B| / |A∪B| = TP / (TP + FP + FN)
    jaccard_score = torch.FloatTensor([i if i>0.65 else 0 for i in jaccard_score])
    jaccard_mean_score = torch.mean(jaccard_score)

    return jaccard_mean_score

def Intersection_over_Union_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_mean_score = torch.mean(iou_score)

    return iou_mean_score

def jaccard_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_score = torch.FloatTensor([i if i>0.65 else 0 for i in iou_score])
    jaccard_threshold = torch.mean(iou_score)

    return jaccard_threshold

# Testing jaccard losses SKY
    
def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred) # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect))/(union - intersect +  1e-7)

def compute_jaccard(y_true, y_pred):
    mean_jaccard = 0.
    thresholded_jaccard = 0.
    for im_index in range(y_pred.shape[0]):
        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard/y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard/y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard

def iou_numpy(labels, outputs):
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + 1e-06 / (union + 1e-06))
    return iou

def jaccard_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    j = (intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection + smooth)
    if (j < 0.65):
        return torch.mean(j)
    return torch.mean(j)

def jaccard_coef_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    j = -(intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection + smooth)
    if (j > 0.65):
        j = j - 1
    return j