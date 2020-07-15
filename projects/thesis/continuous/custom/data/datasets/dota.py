# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np
import cv2
import math

import torch

from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock


from detectron2.data.catalog import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_dota_train_json", "load_dota_test_json"]

DOTA_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 0, "name": "small-vehicle"},
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": 'large-vehicle'},
    {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": 'ship'},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": 'container-crane'},
    {"color": [106, 0, 228], "isthing": 1, "id": 4, "name": 'storage-tank'},
    {"color": [0, 60, 100], "isthing": 1, "id": 5, "name": 'plane'},
    {"color": [0, 80, 100], "isthing": 1, "id": 6, "name": 'tennis-court'},
    {"color": [0, 0, 70], "isthing": 1, "id": 7, "name": 'harbor'},
    {"color": [0, 0, 192], "isthing": 1, "id": 8, "name": 'bridge'},
    {"color": [250, 170, 30], "isthing": 1, "id": 9, "name": 'baseball-diamond'},
    {"color": [100, 170, 30], "isthing": 1, "id": 10, "name": 'roundabout'},
    {"color": [220, 220, 0], "isthing": 1, "id": 11, "name": 'basketball-court'},
    {"color": [175, 116, 175], "isthing": 1, "id": 12, "name": 'swimming-pool'},
    {"color": [250, 0, 30], "isthing": 1, "id": 13, "name": 'soccer-ball-field'},
    {"color": [165, 42, 42], "isthing": 1, "id": 14, "name": 'ground-track-field'},
    {"color": [0, 82, 0], "isthing": 1, "id": 15, "name": "helicopter"},
]
DOTA_CATEGORIES_SUB = [
    {"color": [220, 20, 60], "isthing": 1, "id": 0, "name": "small-vehicle"},
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": 'large-vehicle'},
    {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": 'ship'},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": 'container-crane'},
    {"color": [106, 0, 228], "isthing": 1, "id": 4, "name": 'storage-tank'},
    {"color": [0, 60, 100], "isthing": 1, "id": 5, "name": 'plane'},
]


def register_dota_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_dota_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

def load_dota_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWHA_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts



class DotaAPI:
    def __init__(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        self.features = data['features']

    @staticmethod
    def cvt_dota_to_detectron(dota_bbox: list, patch_size: tuple) -> list:
        """ Processes a coordinate array from a geojson into (cy, cx, height, width, theta) format

        :param (list) coords: an array of shape (N, 8) with 4 corner points of boxes
        :return: (numpy.ndarray) an array of shape (N, 5) with coordinates in proper format
        """
        coord = np.asarray(dota_bbox)
        pts = np.reshape(coord, (-1, 5)).astype(dtype=np.float32)
        cx = pts[:, 0] * patch_size[0]
        cy = pts[:, 1] * patch_size[1]
        width = pts[:, 2] * patch_size[0]
        height = pts[:, 3] * patch_size[1]
        theta = pts[:, 4] * 180 / math.pi

        if width < height:
            width, height = height, width
            theta += 90.0
        arr = [cx, cy, width, height, theta]

        arr = np.asarray(arr).reshape(-1, 5)
        arr = torch.tensor(arr)
        original_dtype = arr.dtype
        arr = arr.double()

        w = arr[:, 2]
        h = arr[:, 3]
        a = arr[:, 4]
        c = torch.abs(torch.cos(a * math.pi / 180.0))
        s = torch.abs(torch.sin(a * math.pi / 180.0))
        # This basically computes the horizontal bounding rectangle of the rotated box
        new_w = c * w + s * h
        new_h = c * h + s * w

        # convert center to top-left corner
        arr[:, 0] -= new_w / 2.0
        arr[:, 1] -= new_h / 2.0
        # bottom-right corner
        arr[:, 2] = arr[:, 0] + new_w
        arr[:, 3] = arr[:, 1] + new_h

        arr = arr[:, :4].to(dtype=original_dtype)
        arr = arr.numpy()
        return arr

    @staticmethod
    def cvt_dota_to_detectron_rotated(dota_bbox: list, patch_size: tuple) -> list:
        """ Processes a coordinate array from a geojson into (cy, cx, height, width, theta) format

        :param (list) coords: an array of shape (N, 8) with 4 corner points of boxes
        :return: (numpy.ndarray) an array of shape (N, 5) with coordinates in proper format
        """
        coord = np.asarray(dota_bbox)
        pts = np.reshape(coord, (-1, 5)).astype(dtype=np.float32)
        cx = pts[:, 0] * patch_size[0]
        cy = pts[:, 1] * patch_size[1]
        width = pts[:, 2] * patch_size[0]
        height = pts[:, 3] * patch_size[1]
        theta = pts[:, 4] * 180 / math.pi

        if width < height:
            width, height = height, width
            theta += 90.0
        detectron_bbox = [cx, cy, width, height, theta]

        return detectron_bbox

def load_dota_test_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        dota_api = DotaAPI(json_file)
        anns = dota_api.features
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    meta = MetadataCatalog.get(dataset_name)

    logger.info("Loaded {} images in dota format from {}".format(len(anns), json_file))

    dataset_dicts = []

    for ann in anns:
        record = {}
        record["file_name"] = os.path.join(image_root, ann['image_id'])
        dataset_dicts.append(record)

    return dataset_dicts


def load_dota_train_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with DACON's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in dota instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        dota_api = DotaAPI(json_file)
        anns = dota_api.features
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    meta = MetadataCatalog.get(dataset_name)
    meta.thing_classes = dota_api.thing_classes


    logger.info("Loaded {} images in dota format from {}".format(len(anns), json_file))

    dataset_dicts = []

    for ann in anns:
        record = {}
        record["file_name"] = os.path.join(image_root, ann['image_id'])
        record["height"] = ann['height']
        record["width"] = ann['width']
        patch_size = (ann['width'], ann['height'])

        objs = []
        properties = ann['properties']
        count = 0
        for p in properties:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            # if int(p["type_id"]) > 6:
            #     continue
            # else:
            count += 1
            obj = {}
            obj["bbox"] = dota_api.cvt_dota_to_detectron(p["bounds_imcoords"].split(","), patch_size)
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            obj["category_id"] = int(p["type_id"])
            objs.append(obj)
        if count == 0:
            continue
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def load_dota_rotated_train_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with DACON's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in dota instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        dota_api = DotaAPI(json_file)
        anns = dota_api.features
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    meta = MetadataCatalog.get(dataset_name)
    meta.thing_classes = dota_api.thing_classes


    logger.info("Loaded {} images in dota format from {}".format(len(anns), json_file))

    dataset_dicts = []

    for ann in anns:
        record = {}
        record["file_name"] = os.path.join(image_root, ann['image_id'])
        record["height"] = ann['height']
        record["width"] = ann['width']
        patch_size = (ann['width'], ann['height'])

        objs = []
        properties = ann['properties']
        count = 0
        for p in properties:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            # if int(p["type_id"]) > 6:
            #     continue
            # else:
            count += 1
            obj = {}
            obj["bbox"] = dota_api.cvt_dota_to_detectron_rotated(p["bounds_imcoords"].split(","), patch_size)
            obj["bbox_mode"] = BoxMode.XYWHA_ABS
            obj["category_id"] = int(p["type_id"])
            objs.append(obj)
        if count == 0:
            continue
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_dota_test_instance(name, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_dota_test_json(json_file, image_root, name))

def register_dota_train_instance(name, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_dota_train_json(json_file, image_root, name))

def register_dota_rotated_train_instance(name, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_dota_rotated_train_json(json_file, image_root, name))

# register_dota_test_instance(
#     "dota_test",
#     '/ws/data/open_datasets/detection/dota/dota_org/train/labels.json',
#     "/ws/data/open_datasets/detection/dota/dota_org/train/images")
# register_dota_train_instance(
#     "dota_train",
#     '/ws/data/open_datasets/detection/dota/dota_org/train/labels.json',
#     "/ws/data/open_datasets/detection/dota/dota_org/train/images")
register_dota_rotated_train_instance(
    "dota_rotated_train",
    '/ws/data/open_datasets/detection/dota/dota_patch_512_256/train/labels.json',
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/train/images")
register_dota_test_instance(
    "dota_rotated_test",
    '/ws/data/open_datasets/detection/dota/dota_patch_512_256/val/labels.json',
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/val/images")



def _get_dota_instances_meta():
    thing_ids = [k["id"] for k in DOTA_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DOTA_CATEGORIES if k["isthing"] == 1]
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DOTA_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_dota_sub_instances_meta():
    thing_ids = [k["id"] for k in DOTA_CATEGORIES_SUB if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DOTA_CATEGORIES_SUB if k["isthing"] == 1]
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DOTA_CATEGORIES_SUB if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

register_dota_instances(
    "dota_rotated_train_coco",
    _get_dota_instances_meta(),
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/train/coco_labels.json",
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/train/images"
)
register_dota_instances(
    "dota_rotated_val_coco",
    _get_dota_instances_meta(),
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/val/coco_labels.json",
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/val/images"
)


register_dota_instances(
    "dota_rotated_train_coco_sub",
    _get_dota_sub_instances_meta(),
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/train/coco_labels_sub.json",
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/train/images"
)
register_dota_instances(
    "dota_rotated_val_coco_sub",
    _get_dota_sub_instances_meta(),
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/val/coco_labels_sub.json",
    "/ws/data/open_datasets/detection/dota/dota_patch_512_256/val/images"
)


def main(args):
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """

    # from continuous.utils.visualizer import Visualizer
    # import detectron2.data.datasets  # noqa # add pre-defined metadata
    #
    #
    # name = "dacon_rotated_all_coco"
    # meta = _get_dacon_instances_meta()
    # json_file = "/ws/data/open_datasets/detection/dacon/dacon_patch_512_256/train/coco_labels_all.json"
    # image_root = "/ws/data/open_datasets/detection/dacon/dacon_patch_512_256/train/images"
    #
    # dicts = load_dacon_json(json_file, image_root, name)
    #
    # for d in dicts:
    #     print(d)
    #     img = np.array(Image.open(d["file_name"]))
    #     visualizer = Visualizer(img, metadata=meta)
    #     vis = visualizer.draw_dataset_dict(d)
    #     image = vis.get_image()
    #     cv2.imshow("test", image)
    #     cv2.waitKey(0)






    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine import default_setup
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import torchvision.transforms as T

    from continuous.custom import add_custom_config
    import detectron2.data.datasets  # noqa # add pre-defined metadata

    from continuous.custom.data.build import build_detection_train_loader
    # from projects.Dacon.rcascade.data import build_detection_train_loader

    def setup(args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        add_custom_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        return cfg

    cfg = setup(args)
    meta = MetadataCatalog.get("dota")

    data_loader = build_detection_train_loader(cfg)
    data_loader_iter = iter(data_loader)

    for batch_data in data_loader_iter:
        for data in batch_data:
            image = data['image']
            image = T.ToPILImage()(image)
            image = np.asarray(image)
            color = (255, 0, 0)

            instance = data['instances']
            pred_boxes = instance.get('gt_boxes').tensor.cpu().detach().numpy()
            scale = 1
            for idx in range(len(instance)):
                pred_box = pred_boxes[idx]

                # start_pt = (int(pred_box[0]), int(pred_box[1]))
                # end_pt = (int(pred_box[2]), int(pred_box[3]))
                # image = cv2.rectangle(image, start_pt, end_pt, color, 2)
                # image = cv2.putText(image, str(pred_class), (int(pred_box[0]), int(pred_box[3])), cv2.FONT_HERSHEY_SIMPLEX,
                #                     0.5, (0, 255, 0), 1)

                points = cv2.boxPoints(((pred_box[0] * scale, pred_box[1] * scale),
                                        (pred_box[2] * scale, pred_box[3] * scale), pred_box[4])).tolist()
                points.append(points[0])
                points = [(int(x), int(y)) for x, y in points]
                image = cv2.polylines(image, np.int32([points]), True, color, 2)
        cv2.imshow("test", image)
        cv2.waitKey(0)



if __name__ == "__main__":
    from detectron2.engine import default_argument_parser
    args = default_argument_parser().parse_args()
    args.config_file = '/ws/external/detectron2/projects/Pretraining/configs/detection/cascade_rs_rcnn_R_50_FPN.yaml'
    main(args)
