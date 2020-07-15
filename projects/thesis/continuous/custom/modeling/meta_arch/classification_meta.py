# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..backbone import build_backbone_pretrain

__all__ = [
    "GeneralizedClassification",
]

@META_ARCH_REGISTRY.register()
class GeneralizedClassification(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.num_classes = cfg.MODEL.CUSTOM.CLASSES

        input_shape = ShapeSpec(channels=num_channels)
        self.backbone = build_backbone_pretrain(cfg, input_shape, self.num_classes)

        self.to(self.device)

        self.criterion = nn.CrossEntropyLoss().cuda(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images, targets = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        classification_loss = self.criterion(features['linear'], targets)

        losses = {}
        losses.update({"classification":classification_loss})
        return losses, features['linear'], targets

    def inference(self, batched_inputs):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images, targets = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        return features

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        targets = []
        for x in batched_inputs:
            image = x[0].to(self.device)
            images.append(self.normalizer(image))
            targets.append(x[1])
        images = torch.stack(images, dim=0).to(self.device)
        targets = torch.tensor(targets).to(self.device)
        return images, targets

