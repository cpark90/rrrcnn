# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from tqdm import tqdm

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data import (
    build_detection_test_loader,
)

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def restore_rotated_patches_with_nms(data_loader_iters, model, image_rootpatch_size=1024, patch_overlay=384, iou_threshold=0.4):
    from detectron2.layers.nms import nms_rotated
    # patch_size = 768
    # patch_overlay = 256
    patch_size = 512
    patch_overlay = 256
    step = patch_size - patch_overlay
    outputs = {}
    print("restore process")
    count = -1
    for idx, data_loader_iter in enumerate(data_loader_iters):
        print("Iterator index: " + str(idx))
        for data in tqdm(data_loader_iter):
            count += 1
            if count > 121 * 20 - 1:
                break
            res = model.inference(data, do_postprocess=True)

            filename = os.path.split(data[0]['file_name'])[1]
            org = os.path.splitext(filename)[0].split('_')[0] + '.png'
            img_idx = int(os.path.splitext(filename)[0].split('_')[1])
            row = int(img_idx / 11)
            col = int(img_idx % 11)

            instance = res[0]['instances']

            pred_boxes = instance.get('pred_boxes').tensor.cpu().detach()
            pred_classes = instance.get('pred_classes').cpu().detach()
            pred_scores = instance.get('scores').cpu().detach()

            for idx in range(len(instance)):
                pred_boxes[idx][0] += step * col
                pred_boxes[idx][1] += step * row

            if not org in outputs.keys():
                outputs[org] = {}
                outputs[org]['pred_boxes'] = []
                outputs[org]['pred_classes'] = []
                outputs[org]['pred_scores'] = []
            outputs[org]['pred_boxes'].append(pred_boxes)
            outputs[org]['pred_classes'].append(pred_classes)
            outputs[org]['pred_scores'].append(pred_scores)

            for key in tqdm(outputs.keys()):
                img_filename = key

    print("nms process")
    for key in tqdm(outputs.keys()):
        if len(outputs[key]['pred_boxes']) == 0:
            continue
        boxes = torch.cat(outputs[key]['pred_boxes'])
        classes = torch.cat(outputs[key]['pred_classes'])
        scores = torch.cat(outputs[key]['pred_scores'])

        result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(classes).cpu().tolist():
            mask = (classes == id).nonzero().view(-1)
            keep = nms_rotated(boxes[mask], scores[mask], iou_threshold)
            result_mask[mask[keep]] = True
        keep = result_mask.nonzero().view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        outputs[key]['pred_boxes'] = boxes[keep]
        outputs[key]['pred_classes'] = classes[keep]
        outputs[key]['pred_scores'] = scores[keep]

    return outputs


def restore_rotated_patch_with_nms(data_loader_iters, model, image_root):#patch_size=1024, patch_overlay=384, iou_threshold=0.4):
    from detectron2.layers.nms import nms_rotated
    import torchvision.transforms.functional as TF
    from PIL import Image
    import numpy as np
    angles = [5 * x for x in range(72)]
    # angles = [0]
    # patch_size = 768
    # patch_overlay = 256
    patch_size = 512
    patch_overlay = 256
    step = patch_size - patch_overlay
    outputs = {}
    print("restore process")
    count = -1
    for idx_o, data_loader_iter in enumerate(data_loader_iters):
        print("Iterator index: " + str(idx_o))
        for data in tqdm(data_loader_iter):
            count += 1
            # 553, 499
            if count != 553:#> 121 * 20 - 1:
                continue
                # break

            # for idx in range(len(data)):
            #     img = data[idx]["image"]
            #     # print(img)
            #     img_pil = TF.to_pil_image(img.detach().cpu()).rotate(angle)
            #     img_tensor = TF.to_tensor(img_pil)
            #     data[idx]["image"] = img_tensor.cuda()

            img_path = os.path.join(image_root, data[0]['file_name'])
            for angle in angles:
                image = cv2.imread(img_path)
                im = Image.fromarray(np.uint8(image)).rotate(angle)
                image = np.array(im)



                data[idx_o]["image"] = torch.tensor(image).permute(2, 0, 1)
                # print(data[idx]["image"])
                res = model.inference(data, do_postprocess=True)

                filename = os.path.split(data[0]['file_name'])[1]
                org = os.path.splitext(filename)[0].split('_')[0] + '.png'
                img_idx = int(os.path.splitext(filename)[0].split('_')[1])
                row = int(img_idx / 11)
                col = int(img_idx % 11)

                instance = res[0]['instances']

                pred_boxes = instance.get('pred_boxes').tensor.cpu().detach()
                pred_classes = instance.get('pred_classes').cpu().detach()
                pred_scores = instance.get('scores').cpu().detach()

                print("visualizing process")
                color = (0, 255, 255)
                print(count)

                pred_boxes = pred_boxes.numpy()
                pred_classes = pred_classes.numpy()
                pred_scores = pred_scores.numpy()
                print(pred_boxes)
                for idx in range(pred_scores.shape[0]):
                    pred_box = pred_boxes[idx]
                    pred_class = pred_classes[idx]
                    score = pred_scores[idx]
                    points = cv2.boxPoints(((pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), pred_box[4])).tolist()
                    points.append(points[0])
                    points = [(int(x), int(y)) for x, y in points]
                    image = cv2.polylines(image, np.int32([points]), True, color, 4)
                    image = cv2.putText(image, str(pred_class), points[0], cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2)

                cv2.imwrite(os.path.join("/ws/output/output_img", "r_" + str(angle) + "_" + filename,), image)

    return outputs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        return DatasetEvaluators([])

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


from continuous.custom import add_custom_config
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg

import cv2
import numpy as np
import torchvision.transforms as T
from continuous.custom.data import TestDatasetMapper512, TestDatasetMapper768, TestDatasetMapper1024

def main(args):
    cfg = setup(args)


    model = Trainer.build_model(cfg)
    # data_loader_512 = build_detection_test_loader(cfg, 'dacon_test', mapper=TestDatasetMapper512)
    # data_loader_768 = build_detection_test_loader(cfg, 'dacon_test', mapper=TestDatasetMapper768)
    # data_loader_1024 = build_detection_test_loader(cfg, 'dacon_test', mapper=TestDatasetMapper1024)
    # data_loader_iters = [iter(data_loader_512), iter(data_loader_768), iter(data_loader_1024)]
    cfg.DATASETS.DATASET_MAPPER = "TestDatasetMapper512"
    data_loader = build_detection_test_loader(cfg, 'dacon_rotated_test')
    data_loader_iters = [iter(data_loader)]
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()

    show = False
    if show:
        outputs = restore_rotated_patches_with_nms(data_loader_iters, model)
        print("result saving")
        dacon = []
        dacon.append(['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y'])
        for key in tqdm(outputs.keys()):
            filename = key
            pred_boxes = outputs[key]['pred_boxes'].numpy()
            pred_classes = outputs[key]['pred_classes'].numpy()
            pred_scores = outputs[key]['pred_scores'].numpy()

            for idx in range(pred_scores.shape[0]):
                pred_box = pred_boxes[idx]
                pred_class = pred_classes[idx]+1
                score = pred_scores[idx]
                points = cv2.boxPoints(((pred_box[0], pred_box[1]),
                                        (pred_box[2], pred_box[3]), pred_box[4])).tolist()

                output = [filename, pred_class, score, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1],
                 points[3][0], points[3][1]]
                output = [str(f) for f in output]
                dacon.append(output)
        dacon = np.asarray(dacon)
        np.savetxt('result_1024_fair_v2.csv', dacon, delimiter=',', fmt='%s')

        # dacon = []
        # dacon.append(['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y'])
        # for data in data_loader_iter:
        #     res = model.inference(data, do_postprocess=True)
        #
        #     filename = os.path.split(data[0]['file_name'])[1]
        #     instance = res[0]['instances']
        #
        #     pred_boxes = instance.get('pred_boxes').tensor.cpu().detach().numpy()
        #     pred_classes = instance.get('pred_classes').cpu().detach().numpy()
        #     pred_scores = instance.get('scores').cpu().detach().numpy()
        #     for idx in range(len(instance)):
        #         pred_box = pred_boxes[idx]
        #         pred_class = pred_classes[idx]
        #         score = pred_scores[idx]
        #         output = [filename, 1, score, pred_box[0], pred_box[1], pred_box[2], pred_box[1], pred_box[2], pred_box[3],
        #          pred_box[0], pred_box[3]]
        #         output = [str(f) for f in output]
        #         outputs.append(output)
        # dacon = np.asarray(dacon)
        # np.savetxt('result.csv', dacon, delimiter=',', fmt='%s')
    else:
        image_root = '/ws/data/open_datasets/detection/dacon/test/images'
        # outputs = restore_rotated_patches_with_nms(data_loader_iters, model)
        outputs = restore_rotated_patch_with_nms(data_loader_iters, model, image_root)
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        # print("visualizing process")
        # for key in tqdm(outputs.keys()):
        #     img_filename = key
        #     img_path = os.path.join(image_root, img_filename)
        #     image = cv2.imread(img_path)
        #     color = (255, 0, 0)
        #
        #     pred_boxes = outputs[key]['pred_boxes'].numpy()
        #     pred_classes = outputs[key]['pred_classes'].numpy()
        #     pred_scores = outputs[key]['pred_scores'].numpy()
        #     for idx in range(pred_scores.shape[0]):
        #         pred_box = pred_boxes[idx]
        #         pred_class = pred_classes[idx]
        #         score = pred_scores[idx]
        #         points = cv2.boxPoints(((pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), pred_box[4])).tolist()
        #         points.append(points[0])
        #         points = [(int(x), int(y)) for x, y in points]
        #         image = cv2.polylines(image, np.int32([points]), True, color, 2)
        #         image = cv2.putText(image, str(pred_class), points[0], cv2.FONT_HERSHEY_SIMPLEX,
        #                             0.5, (0, 255, 0), 2)
        #
        #     cv2.imwrite(os.path.join("/ws/output/output_img", key), image)
            # cv2.imshow("test", image)
            # cv2.waitKey(0)

        # for data in data_loader_iter:
        #     res = model.inference(data, do_postprocess=True)
        #
        #     image = data[0]['image']
        #     image = T.ToPILImage()(image)
        #     image = np.asarray(image)
        #     color = (255, 0, 0)
        #     instance = res[0]
        #     pred_boxes = instance.get('pred_boxes').tensor.cpu().detach().numpy()
        #     pred_classes = instance.get('pred_classes').cpu().detach().numpy()
        #     # pred_scores = instance.get('pred_scores').cpu().detach().numpy()
        #     for idx in range(len(instance)):
        #         pred_box = pred_boxes[idx]
        #         pred_class = pred_classes[idx]
        #         points = cv2.boxPoints(((pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), pred_box[4])).tolist()
        #         points.append(points[0])
        #         points = [(int(x), int(y)) for x, y in points]
        #         image = cv2.polylines(image, np.int32([points]), True, color, 2)
        #         image = cv2.putText(image, str(pred_class), points[0], cv2.FONT_HERSHEY_SIMPLEX,
        #                             0.5, (0, 255, 0), 2)
        #
        #     cv2.imshow("test", image)
        #     cv2.waitKey(0)



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # args.config_file = 'configs/detection/branch_cascade_rs_rcnn_R_50_FPN.yaml'
    args.config_file = 'configs/detection/rs_rcnn_R_50_FPN.yaml'
    # args.config_file = 'configs/detection/cascade_rs_rcnn_R_50_FPN.yaml'
    args.resume = True
    main(args)
