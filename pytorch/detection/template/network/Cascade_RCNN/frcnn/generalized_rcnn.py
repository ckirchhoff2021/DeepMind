# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        assert len(images) == 1, 'Batch size of Cascade RCNN must be 1.'

        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.training:
            detections_1, detector_losses_1 = self.roi_heads[0](features, proposals, images.image_sizes, targets)
            detections_2, detector_losses_2 = self.roi_heads[1](features, detections_1[0], images.image_sizes, targets)
            detections_3, detector_losses_3 = self.roi_heads[2](features, detections_2[0], images.image_sizes, targets)
        else:
            detections_1, detector_losses_1 = self.roi_heads[0](features, proposals, images.image_sizes, targets)
            detections_2, detector_losses_2 = self.roi_heads[1](features, detections_1[0]['cascade_proposals'], images.image_sizes, targets)
            detections_3, detector_losses_3 = self.roi_heads[2](features, detections_2[0]['cascade_proposals'], images.image_sizes, targets)

        
        if not self.training:
            class_logits = (detections_1[0]['class_logits'] + \
                            detections_2[0]['class_logits'] + \
                            detections_3[0]['class_logits']) / 3
            box_regression = detections_3[0]['box_regression']
            proposals = detections_3[0]['proposals']
            boxes, scores, labels = self.roi_heads[2].postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            num_images = len(boxes)
            detections = []
            for i in range(num_images):
                detections.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )

            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        for k in detector_losses_1.keys():
            detector_losses_1[k] += detector_losses_2[k] * 0.5 +  detector_losses_3[k] * 0.25 
        losses.update(detector_losses_1)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections
