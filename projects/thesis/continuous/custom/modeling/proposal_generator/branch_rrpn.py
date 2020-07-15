# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, get_norm
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList, Instances
from detectron2.modeling.proposal_generator.rpn import RPN_HEAD_REGISTRY, build_rpn_head
from detectron2.modeling.anchor_generator import build_anchor_generator

from continuous.custom.modeling import Box2BoxTransform, Box2BoxTransformRotated
from continuous.utils.smooth_l1_loss import smooth_l1_loss
from continuous.custom.continuous import RConv
from .rrpn import CustomizedRRPN

@RPN_HEAD_REGISTRY.register()
class BranchRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4, num_branch: int = 1):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        self.num_branch = num_branch
        kernel_type = 0 if num_branch is 8 else 1
        # 3x3 conv for the hidden representation
        self.rconv = RConv(in_channels, in_channels, kernel_type=kernel_type, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        nn.init.normal_(self.rconv.weight, std=0.01)
        for l in [self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_branch = cfg.MODEL.CUSTOM.BRANCH.NUM_BRANCH
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim, "num_branch": num_branch}

    def forward(self, features: List[torch.Tensor], rot: int = 0):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.rconv(x, rot))
            pred_objectness_logits.append(self.objectness_logits(t))
            x = self.anchor_deltas(t)
            pred_anchor_deltas.append(x)
        return pred_objectness_logits, pred_anchor_deltas




@PROPOSAL_GENERATOR_REGISTRY.register()
class BranchRRPN(CustomizedRRPN):
    """
    Trident RRPN subnetwork.
    """

    def __init__(self, cfg, input_shape):
        super(BranchRRPN, self).__init__(cfg, input_shape)

        self.box2box_transform = Box2BoxTransformRotated(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])
        self.num_branch = cfg.MODEL.CUSTOM.BRANCH.NUM_BRANCH
        self.rotation_direction = 1

    def losses(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes
    ):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Boxes or RotatedBoxes]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_batches = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5)

        num_images = num_batches / self.num_branch
        angle_step = 360.0 / self.num_branch * self.rotation_direction
        gt_anchor_deltas = []
        for branch_idx in range(self.num_branch):
            for image_idx in range(int(num_images)):
                idx = int(branch_idx * num_images + image_idx)
                gt_anchor_deltas.append(self.box2box_transform.get_deltas(anchors, gt_boxes[idx], torch.tensor([branch_idx * angle_step], device=anchors.device)))
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5)

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = (gt_labels == 1) | (gt_labels == -2)
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            self.smooth_l1_beta,
            reduction="sum",
        )

        valid_mask = gt_labels >= 0
        p = torch.sigmoid(cat(pred_objectness_logits, dim=1)[valid_mask])
        gt_target = gt_labels[valid_mask].to(torch.float32)
        ce_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_target,
            reduction="none",
        )
        p_t = p * gt_target + (1 - p) * (1 - gt_target)
        focal_loss = ce_loss * ((1 - p_t) ** self.focal_loss_gamma)
        if self.focal_loss_alpha >= 0:
            alpha_t = self.focal_loss_alpha * gt_target + (1 - self.focal_loss_alpha) * (1 - gt_target)
            objectness_loss = alpha_t * focal_loss
        objectness_loss = objectness_loss.sum()

        normalizer = self.batch_size_per_image * num_images
        return {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }

    def _decode_proposals(self, anchors, pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        num_images = int(N / self.num_branch)
        angle_step = 360.0 / self.num_branch * self.rotation_direction

        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            # Expand anchors to shape (N*Hi*Wi*A, B)

            anchors_i = anchors_i.tensor.unsqueeze(0).expand(num_images, -1, -1).reshape(-1, B)
            proposals_i = []
            for branch_idx in range(self.num_branch):
                start_idx = int(branch_idx * num_images)
                end_idx = start_idx + num_images
                pred_anchor_deltas_i_branch = pred_anchor_deltas_i[start_idx:end_idx]
                pred_anchor_deltas_i_branch = pred_anchor_deltas_i_branch.reshape(-1, B)
                proposals_i.append(self.box2box_transform.apply_deltas(pred_anchor_deltas_i_branch, anchors_i, torch.tensor([branch_idx * angle_step], device=anchors_i.device)))
            proposals_i = torch.cat(proposals_i)

            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        batch_size = images.tensor.shape[0]
        images = ImageList(
            torch.cat([images.tensor] * self.num_branch), images.image_sizes * self.num_branch
        )
        if gt_instances is not None:
            all_gt_instances = []
            step_angle = 360.0 / self.num_branch
            for branch_idx in range(self.num_branch):
                instances = []
                for gt_instance in gt_instances:
                    instance = gt_instance
                    instance.gt_boxes.tensor[:, 4] += self.rotation_direction * step_angle * branch_idx
                    instances.append(instance)
                all_gt_instances.extend(instances)
        else:
            all_gt_instances = None
        gt_instances = all_gt_instances

        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        branch_pred_objectness_logits = []
        branch_pred_anchor_deltas = []
        for rot in range(self.num_branch):
            start_idx = int(rot * batch_size)
            end_idx = int(start_idx + batch_size)
            branch_features = [feature[start_idx:end_idx, :, :, :] for feature in features]
            branch_pred_objectness_logit, branch_pred_anchor_delta = self.rpn_head(branch_features, rot)
            # Transpose the Hi*Wi*A dimension to the middle:
            branch_pred_objectness_logits.append([
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                score.permute(0, 2, 3, 1).flatten(1)
                for score in branch_pred_objectness_logit
            ])
            branch_pred_anchor_deltas.append([
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N, Hi*Wi*A, B)
                x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
                for x in branch_pred_anchor_delta
            ])
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for in_feature_idx in range(len(branch_pred_objectness_logits[0])):
            in_feature_pred_objectness_logits = []
            in_feature_pred_anchor_deltas = []
            for branch_idx in range(self.num_branch):
                in_feature_pred_objectness_logits.append(branch_pred_objectness_logits[branch_idx][in_feature_idx])
                in_feature_pred_anchor_deltas.append(branch_pred_anchor_deltas[branch_idx][in_feature_idx])
            pred_objectness_logits.append(torch.cat(in_feature_pred_objectness_logits))
            pred_anchor_deltas.append(torch.cat(in_feature_pred_anchor_deltas))

        if self.training:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            losses = {k: v * self.loss_weight for k, v in losses.items()}
        else:
            losses = {}

        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

