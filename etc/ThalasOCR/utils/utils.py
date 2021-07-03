
from sys import maxsize
import yaml

from vietocr.tool.predictor import Predictor

import numpy as np
import time
import cv2
# from opencv_draw_annotation import draw_bounding_box
from PIL import Image

from difflib import SequenceMatcher
from pprint import pprint

def resize_image(img, best_size = 800):    
    h, w = img.shape[:2]
    maxhw = max(h, w)
    ratio = best_size / maxhw
    img = cv2.resize(img, None, fx = ratio, fy = ratio)
    h, w = img.shape[:2]
    img = cv2.copyMakeBorder(img, (best_size - h) // 2, (best_size - h) // 2, (best_size - w) // 2, (best_size - w) // 2, cv2.BORDER_CONSTANT,value=[225, 225, 225])
    return img

def similar(a, b, theshold = 0.8):
    return SequenceMatcher(None, a, b).ratio() >= theshold

def cluster_bounding_box(iterable, thresh_hold = 50):
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

def format_ocr_result(vietocr, img, box_texts):
    check_time = time.time()
    boxes = np.array(box_texts, dtype = int)

    top_lefts = np.stack([boxes[:, :, 0].min(axis=1), boxes[:,:,1].min(axis=1)], axis=1)
    bot_rights = np.stack([boxes[:, :, 0].max(axis=1), boxes[:,:,1].max(axis=1)], axis=1)
    boxes = np.concatenate([top_lefts, bot_rights], axis=1).tolist()

    boxes = sorted(boxes)
    boxes = remove_small_box(boxes)

    cluster_box  = dict(enumerate(cluster_bounding_box(boxes), 1))
    cluster_box = remove_small_cluster(cluster_box)

    first_col = sorted(cluster_box[0], key = lambda x : x[1])
    second_col = sorted(cluster_box[1], key = lambda x : x[1])

    # for box in first_col:
    #     draw_bounding_box(img, box, color="red")
    # for box in second_col:
    #     draw_bounding_box(img, box, color="green")
    # cv2.imwrite("checking.jpg", img)

    cell_images = []
    for box in first_col:
        cell_images.append(img[box[1]:box[3], box[0]:box[2]])
    for box in second_col:
        cell_images.append(img[box[1]:box[3], box[0]:box[2]])
    texts = vietocr.batch_predict(cell_images, set_bucket_thresh = 40)
    start_idx_second_col = len(first_col)

    result = []    
    for idx, left_cell in enumerate(first_col):
        right_text = find_right_text(start_idx_second_col, texts, left_cell, second_col)        
        left_text = texts[idx]

        result.append([left_text, right_text])

    print("Time OCR: ", time.time() - check_time)
    return result

def load_model_OCR(cfg='./models/vietocr/config/config.yml', ):
    """Load model VietOCR

    Args:
        cfg (str, optional): [description]. Defaults to './models/vietocr/config/vgg-seq2seq.yml'.

    Returns:
        [type]: [description]
    """

    print("Loading VietOCR......")

    with open('./models/vietocr/config/base.yml', encoding="utf8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    with open(cfg, encoding="utf8") as file:
        config2 = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    
    config.update(config2)

    config['weights'] = './models/vietocr/weights/transformerocr.pth'
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    config['predictor']['beamsearch']=False

    detector = Predictor(config)

    return detector
