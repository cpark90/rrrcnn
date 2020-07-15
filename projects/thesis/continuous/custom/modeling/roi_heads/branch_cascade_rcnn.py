from detectron2.layers import batched_nms_rotated
from detectron2.structures import Instances
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from .cascade_rotated_rcnn import CascadeRROIHeads

def merge_branch_instances(instances, num_branch, nms_thrsh, topk_per_image):
    """
    Merge detection results from different branches of TridentNet.
    Return detection results by applying non-maximum suppression (NMS) on bounding boxes
    and keep the unsuppressed boxes and other instances (e.g mask) if any.

    Args:
        instances (list[Instances]): A list of N * num_branch instances that store detection
            results. Contain N images and each image has num_branch instances.
        num_branch (int): Number of branches used for merging detection results for each image.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        results: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections after merging results from multiple
            branches.
    """
    batch_size = len(instances) // num_branch
    results = []
    for i in range(batch_size):
        ins = []
        for j in range(num_branch):
            ins.append(instances[i + batch_size * j])
        instance = Instances.cat(ins)

        # Apply per-class NMS
        keep = batched_nms_rotated(
            instance.pred_boxes.tensor, instance.scores, instance.pred_classes, nms_thrsh
        )
        keep = keep[:topk_per_image]
        result = instance[keep]

        results.append(result)

    return results

@ROI_HEADS_REGISTRY.register()
class BranchROIHeads(CascadeROIHeads):
    """
    The `CascadeRROIHeads` for TridentNet.
    See :class:`TCascadeRROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super(BranchROIHeads, self).__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.CUSTOM.BRANCH.NUM_BRANCH
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`TridentCRROIHeads.forward`.
        """
        num_branch = self.num_branch
        # Duplicate images and gt_instances for all branches in TridentNet.
        all_targets = targets * num_branch if targets is not None else None

        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances, num_branch, self.test_nms_thresh, self.test_detections_per_img
            )

            return pred_instances, {}

@ROI_HEADS_REGISTRY.register()
class BranchRROIHeads(CascadeRROIHeads):
    """
    The `CascadeRROIHeads` for TridentNet.
    See :class:`TCascadeRROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super(BranchRROIHeads, self).__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.CUSTOM.BRANCH.NUM_BRANCH
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`TridentCRROIHeads.forward`.
        """
        num_branch = self.num_branch
        # Duplicate images and gt_instances for all branches in TridentNet.
        all_targets = targets * num_branch if targets is not None else None

        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances, num_branch, self.test_nms_thresh, self.test_detections_per_img
            )

            return pred_instances, {}

