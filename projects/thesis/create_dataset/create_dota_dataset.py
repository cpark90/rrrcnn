"""Create TFRecord from geojson """

import os
import re
import json
import argparse
import math
from collections import namedtuple
from shapely.geometry import Polygon
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2

# Object Class
Object = namedtuple('Object', 'coord cls_idx cls_text')

# Pacth Class
Patch = namedtuple('Patch', 'image_id image row col objects')

META_DATA = {
    'small-vehicle': 0,
    'large-vehicle': 1,
    'ship': 2,
    'container-crane': 3,
    'storage-tank': 4,
    'plane': 5,
    'helicopter': 6,
    'tennis-court': 7,
    'harbor': 8,
    'bridge': 9,
    'baseball-diamond': 10,
    'roundabout': 11,
    'basketball-court': 12,
    'swimming-pool': 13,
    'soccer-ball-field': 14,
    'ground-track-field': 15,
}

def restore_patches_with_nms(data_loader_iter, model, patch_size=768, patch_overlay=256, nms_thresh=0.5):
    step = patch_size - patch_overlay
    for data in data_loader_iter:
        res = model.inference(data, do_postprocess=True)

        filename = os.path.split(os.path.split(data[0]['file_name'])[1])[0]
        instance = res[0]['instances']

        pred_boxes = instance.get('pred_boxes').tensor.cpu().detach().numpy()
        pred_classes = instance.get('pred_classes').cpu().detach().numpy()
        pred_scores = instance.get('scores').cpu().detach().numpy()


def get_patch_image(image, row, col, patch_size):
    patch_image_height = patch_size if image.shape[0] - row > patch_size else image.shape[0] - row
    patch_image_width = patch_size if image.shape[1] - col > patch_size else image.shape[1] - col

    patch_image = image[row: row + patch_image_height, col: col + patch_image_width]

    if patch_image_height < patch_size or patch_image_width < patch_size:
        pad_height = patch_size - patch_image_height
        pad_width = patch_size - patch_image_width
        patch_image = np.pad(patch_image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

    return patch_image

def cvt_coords_to_rboxes(coords):
    """ Processes a coordinate array from a geojson into (cy, cx, height, width, theta) format

    :param (numpy.ndarray) coords: an array of shape (N, 8) with 4 corner points of boxes
    :return: (numpy.ndarray) an array of shape (N, 5) with coordinates in proper format
    """

    rboxes = []
    for coord in coords:
        pts = np.reshape(coord, (-1, 2)).astype(dtype=np.float32)
        (cx, cy), (width, height), theta = cv2.minAreaRect(pts)

        if width < height:
            width, height = height, width
            theta += 90.0
        rboxes.append([cy, cx, height, width, math.radians(theta)])

    return np.array(rboxes)


def cvt_coords_to_polys(coords):
    """ Convert a coordinate array from a geojson into Polygons

    :param (numpy.ndarray) coords: an array of shape (N, 8) with 4 corner points of boxes
    :return: (numpy.ndarray) polygons: an array of shapely.geometry.Polygon corresponding to coords
    """

    polygons = []
    for coord in coords:
        polygons.append(Polygon([coord[0:2], coord[2:4], coord[4:6], coord[6:8]]))
    return np.array(polygons)

def IoA(poly1, poly2):
    """ Intersection-over-area (ioa) between two boxes poly1 and poly2 is defined as their intersection area over
    box2's area. Note that ioa is not symmetric, that is, IOA(poly1, poly2) != IOA(poly1, poly2).

    :param (shapely.geometry.Polygon) poly1: Polygon1
    :param (shapely.geometry.Polygon) poly2: Polygon2
    :return: (float) IoA between poly1 and poly2
    """
    return poly1.intersection(poly2).area / poly1.area




def load_test_geojson(filename):
    """ Gets label data from a geojson label file

    :param (str) filename: file path to a geojson label file
    :return: (numpy.ndarray, numpy.ndarray ,numpy.ndarray) coords, chips, and classes corresponding to
            the coordinates, image names, and class codes for each ground truth.
    """

    with open(filename) as f:
        data = json.load(f)

    image_ids = np.zeros((len(data['features'])), dtype='object')

    for idx in range(len(data['features'])):
        properties = data['features'][idx]['properties']
        image_ids[idx] = properties['image_id']

    return image_ids


def save_test_patches(imgs_dst, patches):
    ##save images and append annotation
    features = []
    for idx, patch in enumerate(patches):

        image = patch.image
        org_name = os.path.splitext(patch.image_id)[0]
        img_name = os.path.join(org_name, org_name + '_' + str(idx)) + '.png'

        cv2.imwrite(os.path.join(imgs_dst, img_name), image)

        patch_height = patch.image.shape[0]
        patch_width = patch.image.shape[1]

        feature = {}
        feature['image_id'] = img_name
        feature['width'] = patch_width
        feature['height'] = patch_height
        features.append(feature)

    return features


def create_test_dataset(src_dir, dst_path, patch_size=1024, patch_overlay=384, object_fraction_thresh=0.7,
                     is_include_only_pos=False):
    """ Create TF Records from geojson

    :param (str) src_dir: path to a GeoJson file
    :param (str) dst_path: Path to save data'
    :param (int) patch_size: patch size
    :param (int) patch_overlay: overlay size for patching
    :param (float) object_fraction_thresh: threshold value for determining contained objects
    :param (bool) is_include_only_pos: Whether or not to include only positive patch image(containing at least one object)
    :return:
    """

    n_tfrecord = 0

    # Load objects from geojson
    geojson_path = os.path.join(src_dir, 'labels.json')
    image_ids = load_test_geojson(geojson_path)


    json_dst = os.path.join(dst_path, 'labels.json')
    imgs_dst = os.path.join(dst_path, 'images')
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    if not os.path.isdir(imgs_dst):
        os.mkdir(imgs_dst)

    features = {'features': []}

    # Load image files as TIF
    for image_id in tqdm(sorted(set(image_ids))):

        image = cv2.imread(os.path.join(src_dir, 'images/', image_id))

        # Create patches including objects
        patches = []
        step = patch_size - patch_overlay
        for row in range(0, image.shape[0] - patch_overlay, step):
            for col in range(0, image.shape[1] - patch_overlay, step):
                # Check if a patch contains objects and append objects
                objects_in_patch = []
                patch_image = get_patch_image(image, row, col, patch_size)

                patches.append(
                    Patch(image_id=image_id, image=patch_image, row=row, col=col, objects=objects_in_patch))

        img_dst = os.path.join(imgs_dst, os.path.splitext(image_id)[0])
        if not os.path.isdir(img_dst):
            os.mkdir(img_dst)
        features['features'] += save_test_patches(imgs_dst, patches)

        n_tfrecord += len(patches)
    with open(json_dst, 'w') as f:
        json.dump(features, f)

    print('N of TFRecords:', n_tfrecord)


def load_data(filename):
    """ Gets label data from a geojson label file

    :param (str) filename: file path to a geojson label file
    :return: (numpy.ndarray, numpy.ndarray ,numpy.ndarray) coords, chips, and classes corresponding to
            the coordinates, image names, and class codes for each ground truth.
    """
    imgs_src = os.path.join(filename, 'images')
    anns_src = os.path.join(filename, 'annotations')

    image_names = os.listdir(imgs_src)

    data_len = len(image_names)

    obj_coords = []
    image_ids = []
    class_indices = []
    class_names = []

    for image_idx in range(data_len):
        image_name, ext = os.path.splitext(image_names[image_idx])
        if ext != '.png':
            continue
        # image_id = int(re.search(r'\d+', image_name).group())
        try:
            annotations = pd.read_csv(os.path.join(anns_src, image_name + '.txt'), sep=' ', skiprows=2)
        except:
            print('No annotations in file name: ' + image_name)
            continue
        for idx in range(annotations.shape[0]):
            image_ids.append(image_names[image_idx])
            obj_coord = annotations.iloc[idx, :8].to_numpy()
            class_index = META_DATA[annotations.iloc[idx, 8]]
            class_name = annotations.iloc[idx, 8]

            obj_coords.append(obj_coord)
            class_indices.append(class_index)
            class_names.append(class_name)

    obj_coords = np.asarray(obj_coords)
    image_ids = np.asarray(image_ids, dtype='object')
    class_indices = np.asarray(class_indices, dtype=int)
    class_names = np.asarray(class_names, dtype='object')

    return image_ids, obj_coords, class_indices, class_names


def save_patches(imgs_dst, patches):
    ##save images and append annotation
    features = []
    for idx, patch in enumerate(patches):

        image = patch.image
        org_name = os.path.splitext(patch.image_id)[0]
        img_name = os.path.join(org_name, org_name + '_' + str(idx)) + '.png'

        cv2.imwrite(os.path.join(imgs_dst, img_name), image)

        patch_height = patch.image.shape[0]
        patch_width = patch.image.shape[1]

        feature = {}
        feature['image_id'] = img_name
        feature['width'] = patch_width
        feature['height'] = patch_height
        feature['properties'] = []
        for coord, cls_idx, cls_text in patch.objects:
            properties = {}

            center_xs = coord[1] / patch_width
            center_ys = coord[0] / patch_height
            height = coord[2] / patch_height
            width = coord[3] / patch_width
            theta = coord[4]

            properties['bounds_imcoords'] = str(center_xs) + ',' + str(center_ys) + ',' + str(width) + ',' + str(height) + ',' + str(theta)
            properties['type_id'] = int(cls_idx)
            feature['properties'].append(properties)
        features.append(feature)

    return features


def create_train_dataset(src_dir, dst_path, patch_size=1024, patch_overlay=384, object_fraction_thresh=0.7,
                     is_include_only_pos=False):
    """ Create TF Records from geojson

    :param (str) src_dir: path to a GeoJson file
    :param (str) dst_path: Path to save data'
    :param (int) patch_size: patch size
    :param (int) patch_overlay: overlay size for patching
    :param (float) object_fraction_thresh: threshold value for determining contained objects
    :param (bool) is_include_only_pos: Whether or not to include only positive patch image(containing at least one object)
    :return:
    """

    n_tfrecord = 0

    # Load objects from geojson
    image_ids, obj_coords, class_indices, class_names = load_data(src_dir)

    obj_polys = cvt_coords_to_polys(obj_coords)
    obj_coords = cvt_coords_to_rboxes(obj_coords)

    json_dst = os.path.join(dst_path, 'labels.json')
    imgs_dst = os.path.join(dst_path, 'images')
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    if not os.path.isdir(imgs_dst):
        os.mkdir(imgs_dst)

    features = {'features': []}

    # Load image files as TIF
    for image_id in tqdm(sorted(set(image_ids))):

        image = cv2.imread(os.path.join(src_dir, 'images/', image_id))

        # Get data in the current image
        obj_coords_in_image = obj_coords[image_ids == image_id]
        obj_polys_in_image = obj_polys[image_ids == image_id]
        class_indices_in_image = class_indices[image_ids == image_id]
        class_texts_in_image = class_names[image_ids == image_id]

        # Create patches including objects
        patches = []
        step = patch_size - patch_overlay
        for row in range(0, image.shape[0] - patch_overlay, step):
            for col in range(0, image.shape[1] - patch_overlay, step):
                patch_poly = Polygon([(col, row), (col + patch_size, row),
                                      (col + patch_size, row + patch_size), (col, row + patch_size)])

                # Check if a patch contains objects and append objects
                objects_in_patch = []
                for idx, obj_poly in enumerate(obj_polys_in_image):
                    if IoA(obj_poly, patch_poly) > object_fraction_thresh:
                        objects_in_patch.append(Object(obj_coords_in_image[idx], class_indices_in_image[idx],
                                                       class_texts_in_image[idx]))

                # if a patch contains objects, append the patch to save tfrecords
                if not is_include_only_pos or objects_in_patch:
                    objects_in_patch = [
                        Object(coord=[obj.coord[0] - row, obj.coord[1] - col, obj.coord[2], obj.coord[3], obj.coord[4]],
                               cls_idx=obj.cls_idx, cls_text=obj.cls_text) for obj in objects_in_patch]
                    patch_image = get_patch_image(image, row, col, patch_size)

                    patches.append(
                        Patch(image_id=image_id, image=patch_image, row=row, col=col, objects=objects_in_patch))

        img_dst = os.path.join(imgs_dst, os.path.splitext(image_id)[0])
        if not os.path.isdir(img_dst):
            os.mkdir(img_dst)
        features['features'] += save_patches(imgs_dst, patches)

        n_tfrecord += len(patches)
    with open(json_dst, 'w') as f:
        json.dump(features, f)

    print('N of TFRecords:', n_tfrecord)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create patches from geojson')
    parser.add_argument('--src_dir',
                        type=str,
                        # required=True,
                        metavar='DIR',
                        help='Root directory to geojson and images')
    parser.add_argument('--dst_path',
                        type=str,
                        metavar='FILE',
                        help='Path to save patches')
    parser.add_argument('--patch_size',
                        type=int,
                        default=512,
                        help='Patch size')
    parser.add_argument('--patch_overlay',
                        type=int,
                        default=256,
                        help='Overlay size for patching')
    parser.add_argument('--object_fraction_thresh',
                        type=float,
                        default=0.7,
                        help='Threshold value for determining contained objects')
    parser.add_argument('--is_include_only_pos',
                        dest='is_include_only_pos',
                        action='store_true',
                        help='Whether or not to include only positive patch image(containing at least one object)')

    args = parser.parse_args()

    args.src_dir = '/ws/data/open_datasets/detection/dota/dota_org/val'
    args.dst_path = '/ws/data/open_datasets/detection/dota/dota_patch_512_256/val'
    args.is_include_only_pos = True

    create_train_dataset(**vars(args))

    # args.src_dir = '/ws/data/open_datasets/detection/dota/dota_org/val'
    # args.dst_path = '/ws/data/open_datasets/detection/dota/dota_patch_512_256/val'
    # args.is_include_only_pos = True
    #
    # create_test_dataset(**vars(args))
