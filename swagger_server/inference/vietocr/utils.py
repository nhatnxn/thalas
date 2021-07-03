import yaml
from vietocr.tool.predictor import Predictor
from difflib import SequenceMatcher
import torch
import numpy as np
from PIL import Image
import math
from collections import defaultdict
import cv2


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def remove_small_cluster(cluster_box, thresh_area = 400):
    good_cluster = []
    for cluster in cluster_box:
        max_area = 0
        boxes = cluster_box[cluster]
        for box in boxes:
            max_area = max(max_area, (box[2] - box[0]) * (box[3] - box[1]))
        if max_area >= thresh_area:
            good_cluster.append(cluster_box[cluster])
    return good_cluster


def find_right_text(start_idx_second_col, texts, left_cell, second_col):
    for idx, right_cell in enumerate(second_col):
        center_cell_y = (left_cell[1] + left_cell[3]) / 2
        if right_cell[1] <= center_cell_y <= right_cell[3]:
            return texts[start_idx_second_col + idx]
    return ""


def remove_small_box(boxes, threshold_x = 11, threshold_y = 11):
    good_boxes = []
    for box in boxes:
        if box[2] - box[0] <= threshold_x or box[3] - box[1] <= threshold_y:
            continue
        else:
            good_boxes.append(box)
    return good_boxes


def cluster_bounding_box(iterable, thresh_hold=50):
    prev = None
    group = []
    for item in iterable:
        if not prev or item[0] - prev[0] <= thresh_hold:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def similar(a, b, theshold = 0.8):
    return SequenceMatcher(None, a, b).ratio() >= theshold


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


def sort_width(batch_img: list, reverse: bool = False):
    """
    Sort list image correspondint to width of each image

    Parameters
    ----------
    batch_img: list
        list input image

    Return
    ------
    sorted_batch_img: list
        sorted input images
    width_img_list: list
        list of width images
    indices: list
        sorted position of each image in original batch images
    """
    def get_img_width(element):
        img = element[0]
        c, h, w = img.shape
        return w

    batch = list(zip(batch_img, range(len(batch_img))))
    sorted_batch = sorted(batch, key=get_img_width, reverse=reverse)
    sorted_batch_img, indices = list(zip(*sorted_batch))
    width_img_list = list(map(get_img_width, batch))

    return sorted_batch_img, width_img_list, indices


def resize_v1(w: int, h: int, expected_height: int, image_min_width: int, image_max_width: int):
    """
    Get expected height and width of image

    Parameters
    ----------
    w: int
        width of image
    h: int
        height
    expected_height: int
    image_min_width: int
    image_max_width: int
        max_width of

    Return
    ------

    """
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def preprocess_input(image):
    h, w, _ = image.shape
    new_w, image_height = resize_v1(w, h, 32, 32, 512)

    img = cv2.resize(image, (new_w, image_height))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))

    return img


def resize_v2(img, width, height):
    """
    Resize bucket images into fixed size to predict on  batch size
    """
    new_img = np.transpose(img, (1, 2, 0))
    new_img = cv2.resize(new_img, (width, height), cv2.INTER_AREA)
    new_img = np.transpose(new_img, (2, 0, 1))

    return new_img


def batch_process(images, set_bucket_thresh):
    batch_img_dict = defaultdict(list)
    image_height = 32
    img_li = [preprocess_input(img) for img in images]
    img_li, width_list, indices = sort_width(img_li, reverse=False)
    min_bucket_width = min(width_list)
    max_width = max(width_list)
    max_bucket_width = np.minimum(
        min_bucket_width + set_bucket_thresh, max_width)
    for i, image in enumerate(img_li):
        c, h, w = image.shape
        if w > max_bucket_width:
            min_bucket_width = w
            max_bucket_width = np.minimum(
                min_bucket_width + set_bucket_thresh, max_width)
        avg_bucket_width = int((max_bucket_width + min_bucket_width) / 2)
        new_img = resize_v2(
            image, avg_bucket_width, height=image_height)
        batch_img_dict[avg_bucket_width].append(new_img)
    return batch_img_dict, indices