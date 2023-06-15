# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn


class PoseAuxLoss():
    def __init__(self, reduction):
        self.reduction = reduction

    def __call__(self, logits, learned_masks, gt_labels, gt_pose_mask, scale=1):
        """
        **Arguments**
        logits : type
            The logits predicted by the model
        learned_mask : list[torch.Tensor]
            The attention mask from the poseblock
        gt_labels : type
            The true labels
        gt_pose_mask : type
            The true pose mask
        """
        classification_loss = nn.CrossEntropyLoss(reduction=self.reduction)(logits, gt_labels)

        mask_loss = 0
        num_masks = len(learned_masks)

        for learned_mask in learned_masks:
            mask_loss = mask_loss + nn.BCELoss(reduction=self.reduction)(learned_mask, gt_pose_mask)

        mask_loss = mask_loss / num_masks

        return (classification_loss) + (scale*mask_loss)

# def pose_auxiliary_loss(logits, learned_masks, gt_labels, gt_pose_mask):
#     """
#     **Arguments**
#     logits : type
#         The logits predicted by the model
#     learned_mask : list[torch.Tensor]
#         The attention mask from the poseblock
#     gt_labels : type
#         The true labels
#     gt_pose_mask : type
#         The true pose mask
#     """
#     classification_loss = nn.BCELoss()(logits, gt_labels)

#     mask_loss = 0
#     num_masks = len(learned_masks)

#     for learned_mask in learned_masks:
#         mask_loss = mask_loss + nn.BCELoss()(learned_mask, gt_pose_mask)

#     mask_loss = mask_loss / num_masks

#     return classification_loss + mask_loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "posemask_loss": PoseAuxLoss
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
