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


DACON_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 0, "name": "Container"},
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": 'Oil'},
    # {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": 'Carrier'},
    {"color": [0, 0, 230], "isthing": 1, "id": 2, "name": 'ETC'},
]



class DaconAPI:
    def __init__(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        self.features = data['features']

    @staticmethod
    def cvt_dacon_to_detectron(dota_bbox: list, patch_size: tuple) -> list:
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
    def cvt_dacon_to_detectron_rotated(dota_bbox: list, patch_size: tuple) -> list:
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

def main():
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

    data_path = "/ws/data/open_datasets/detection/dacon/dacon_patch_512_256/train"
    json_file = os.path.join(data_path, 'labels.json')
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        dacon_api = DaconAPI(json_file)
        anns = dacon_api.features

    dataset_dicts = []
    for ann in anns:
        record = {}
        record["file_name"] = ann['image_id']
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
            count += 1
            obj = {}
            obj["bbox"] = dacon_api.cvt_dacon_to_detectron_rotated(p["bounds_imcoords"].split(","), patch_size)
            obj["bbox_mode"] = BoxMode.XYWHA_ABS
            if int(p["type_id"]) == 4:
                obj["category_id"] = 3
            else:
                obj["category_id"] = int(p["type_id"])
            objs.append(obj)
        if count == 0:
            continue
        record["annotations"] = objs
        dataset_dicts.append(record)

    output_dict = {"type":"instances","images":[],"annotations":[],
                   "categories": [
                       {
                           "supercategory": "none",
                           "name": d["name"],
                           "id": d["id"]
                       } for d in DACON_CATEGORIES
                   ]}
    for record in dataset_dicts:
        image = {}
        image["file_name"] = record["file_name"]
        image["height"] = record["height"]
        image["width"] = record["width"]
        f, b = os.path.splitext(os.path.split(record["file_name"])[1])[0].split('_')
        f = int(''.join(i for i in f if i.isdigit()))
        b = int(''.join(i for i in b if i.isdigit()))
        # if f < 2500:
        #     continue

        image_id = f * 1000 + b
        image["id"] = image_id
        output_dict["images"].append(image)

        count = 0
        for obj in record["annotations"]:
            annotation = {}
            annotation["id"] = image_id * 10000 + count
            bbox = [d.item() for d in obj["bbox"]]
            annotation["bbox"] = bbox
            annotation["image_id"] = image_id
            annotation["ignore"] = 0
            annotation["area"] = bbox[2] * bbox[3]
            annotation["iscrowd"] = 0
            annotation["category_id"] = obj["category_id"] - 1
            output_dict["annotations"].append(annotation)
            count += 1

    output_path = os.path.join(data_path, "coco_labels_all.json")
    with open(output_path, 'w') as outfile:
        json.dump(output_dict, outfile)


main()