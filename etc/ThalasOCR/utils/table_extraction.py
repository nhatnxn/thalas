import cv2
import numpy as np
import os
from collections import defaultdict
import string
from openpyxl import Workbook
from PIL import Image
import torch
from torch.autograd import Variable
from . import craft_utils, imgproc

def resize_img(img, SIZE = 2048):
    max_wh = max(img.shape)
    ratio = SIZE/max_wh
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    return img

def min_edge_resize(img, FIX_WIDTH = 640):
    """
    assume img shape [height, width, channel]
    """
    w = img.shape[1]

    ratio = FIX_WIDTH/w
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    return img


def invert_img(img):
    '''
    return binary image with white lines, text and black background
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 9, 15, 15) 
    not_image = cv2.bitwise_not(img_gray)
    img_bin = cv2.adaptiveThreshold(not_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -4)
    return img_bin

def get_horizontal_lines(img_bin):
    '''
    return horizontal lines
    '''
    kernel_length_h = 10
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)
    return horizontal_lines_img

def get_vertical_lines(img_bin):
    '''
    return vertical lines
    '''
    kernel_length_v = 10
    #create kernel
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)
    return vertical_lines_img

def find_cell_bbox(contours, hierarchy):
    list_cell = []
    
    h_list = []
    color = (255, 0, 0)
    thickness = 2
    for idx, c in enumerate(contours):
        xc, yc, wc, hc = cv2.boundingRect(c)
        if (wc > 10 and hc > 10) and (hierarchy[0][idx][2] == -1) and (hierarchy[0][idx][3] != -1):
            h_list.append(hc)
            list_cell.append([xc, yc, xc + wc, yc + hc])
    h_mean = sum(h_list)/len(h_list)
    return list_cell, h_mean

def joints_ver_hor_lines(vertical_lines_img, horizontal_lines_img):
    '''
    return joints vertical and horizontal lines
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ## add weight thuc hien viec cong hai anh theo weights
    table_segment = cv2.addWeighted(vertical_lines_img, 1, horizontal_lines_img, 1, 0.0)
    ## thuc hien dao nguoc anh va lam nho nhung vung mau trang ==> duong net mau den se hien ra to hon
    horizontal_lines_img = cv2.dilate(table_segment, kernel, iterations=3)    
    return horizontal_lines_img


def draw_cell_lines(cell_img, horizontal_boxes, vertical_boxes):
    '''
    return table image
    '''
    #Draw horizontal lines
    for box in horizontal_boxes:
        x,y,w,h = box
        start_x = x
        start_y = int(y+h/2)
        end_x = x+w
        end_y = start_y
        cell_img = cv2.line(cell_img, (start_x, start_y), (end_x, end_y), (255,255,255), 1)

    #Draw vertical lines
    for box in vertical_boxes:
        x,y,w,h = box
        start_x = int(x + w/2)
        start_y = y - 5
        end_x = start_x
        end_y = y + h + 3
        cell_img = cv2.line(cell_img, (start_x, start_y), (end_x, end_y), (255,255,255), 1)
    return cell_img

def get_row_col_dict(cell_boxes):
    '''
    return row (dict) and column (dict) coordinate
    '''
    row = get_row_dict(cell_boxes)
    col = get_col_dict(cell_boxes, max_columns=2)
    return row, col


def get_col_dict(list_cell, max_columns=0):
    '''
    return col_dict = {'A': 51, 'B': 334, 'C': 1064}
    '''
    col_dict = defaultdict()
    col_count = 0
    list_cell = sorted(list_cell, key=lambda k: [k[0], k[1]])
    for i in range(len(list_cell)-2):
        x = list_cell[i][0]
        x_next = list_cell[i+1][0]
        if x_next-x > 30:
            col = int((x_next+x)/2)
            if col_count>25:
                col_dict[string.ascii_uppercase[col_count//26-1]+string.ascii_uppercase[col_count%26-26]] = col
            else:
                col_dict[string.ascii_uppercase[col_count]] = col
            col_count += 1
    # #add 1 last col
    col = int((list_cell[-1][0] + list_cell[-1][2])/2)
    if col_count>25:
        col_dict[string.ascii_uppercase[col_count//26-1]+string.ascii_uppercase[col_count%26-26]] = col
    else:
        col_dict[string.ascii_uppercase[col_count]] = col

    ks = col_dict.keys()
    if max_columns > 0 and len(ks) > max_columns:
        filtered_col_dict = {}
        for i, k in enumerate(ks):
            if i < max_columns:
                filtered_col_dict[k] = col_dict[k]
        
        col_dict= filtered_col_dict

    return col_dict

def get_row_dict(list_cell):
    row_dict = defaultdict()
    row_count = 1
    list_cell = sorted(list_cell, key=lambda k: [k[1], k[0]])
    for i in range(len(list_cell)-2):
        y = list_cell[i][1]
        y_next = list_cell[i+1][1]
        if y_next-y > 20:
            row = int((y_next+y)/2)
            row_dict[str(row_count)] = row
            row_count += 1
    # # add 1 last row
    if (list_cell[-1][3] - list_cell[-1][1]) > 15:
        row_dict[str(row_count)] = int((list_cell[-1][1] + list_cell[-1][3])/2)
    else:
        row_dict[str(row_count)] = int((list_cell[-2][1] + list_cell[-2][3])/2)

    return row_dict


def write_sheet(sheet, sheet_cell_dict, list_sheet_merge):

    for cell, text in sheet_cell_dict.items():
        sheet[cell] = text
    for merge_range in list_sheet_merge:
        sheet.merge_cells(merge_range)
    return sheet


def text_detection(net, image, text_threshold = 0.3, link_threshold = 0.3, low_text = 0.3, cuda = True, poly = False, refine_net=None):
    canvas_size = 2048
    mag_ratio = 1.5
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    return boxes


def get_sheet_cell_and_merge(detector, list_cell, box_text_by_cell, row_dict, col_dict, img):
    ''' detector: model OCR
        list_cell: [[x1, y1, x2, y2] list tọa độ các box top left, bottom right của cell
        box_text_by_cell: {'x1, y1, x2, y2': [([xmin, ymin, xmax, ymax], ([xmin, ymin, xmax, ymax]]} key là tọa độ của cell, value là 
                        list các box nằm trong cell
        row_dict: {'0': 23.4, '1': 56.7} key là tên hàng, value là tọa độ theo y của đường trung trực đi qua 1 cell
        col_dict: {'A': 45.6, 'B': 109.5} tương tự như trên
        img: ảnh ban đầu
    '''
        
    sheet_cell_dict = defaultdict()
    list_sheet_merge = []
    for cell in list_cell:
        x1, y1, x2, y2 = cell

        # check cell in colums:
        in_columns = False
        for col, col_x in col_dict.items():
            if x1 < col_x and x2 > col_x:
                in_columns = True
                break

        if not in_columns:
            continue
        
        list_boxes = box_text_by_cell[str(cell)]
        text = ''
        if len(list_boxes) == 0:
            pass
        elif len(list_boxes) == 1:
            xmin = list_boxes[0][0]
            ymin = list_boxes[0][1]
            xmax = list_boxes[0][2]
            ymax = list_boxes[0][3]
            
            cell_image = img[ymin:ymax, xmin:xmax]
            cell_image = Image.fromarray(cell_image)
            text = detector.predict(cell_image)
        else:
            list_boxes = sorted(list_boxes, key=lambda k: [k[1], k[0]])
            
            if (np.array(list_boxes)[:, 1].max() - np.array(list_boxes)[:, 1].min()) < 15:
                xmin = np.array(list_boxes)[:, 0].min()
                ymin = np.array(list_boxes)[:, 1].min()
                xmax = np.array(list_boxes)[:, 2].max()
                ymax = np.array(list_boxes)[:, 3].max()
                
                cell_image = img[ymin:ymax, xmin:xmax]
                cell_image = Image.fromarray(cell_image)
                text+= " " + detector.predict(cell_image)
            else:
                same_line = [list_boxes[0]]
                xmin = np.array(same_line)[:, 0].min()
                ymin = np.array(same_line)[:, 1].min()
                xmax = np.array(same_line)[:, 2].max()
                ymax = np.array(same_line)[:, 3].max()
                for i, box in enumerate(list_boxes[:-1]):
                    
                    if abs(list_boxes[i][1] - list_boxes[i+1][1]) < 15:
                        same_line.append(list_boxes[i+1])
                       
                    else:
                        cell_image = img[ymin:ymax, xmin:xmax]
                        cell_image = Image.fromarray(cell_image)
                        text+= " " + detector.predict(cell_image)
                        same_line = [list_boxes[i+1]]
    
                    xmin = np.array(same_line)[:, 0].min()
                    ymin = np.array(same_line)[:, 1].min()
                    xmax = np.array(same_line)[:, 2].max()
                    ymax = np.array(same_line)[:, 3].max()
                    
                cell_image = img[ymin:ymax, xmin:xmax]
                cell_image = Image.fromarray(cell_image)
                text+= " " + detector.predict(cell_image)
        
        text = text.strip()
        merge_row = []
        merge_col = []
        merge_count = 0
        merge_start = ''
        merge_end = ''
        for row, row_y in row_dict.items():
            if y2 < row_y:
                break
            if y1 < row_y and y2 > row_y:
                merge_row.append(row)
            
        for col, col_x in col_dict.items():
            if x2 < col_x:
                break
            if x1 < col_x and x2 > col_x:
                merge_col.append(col)
        if len(merge_row) == 0 or len(merge_col) == 0:
            continue
            
        sheet_cell_dict[merge_col[0] + merge_row[0]] = text
        if len(merge_row) > 1 or len(merge_col) > 1:
            merge_start = merge_col[0] + merge_row[0]
            merge_end = merge_col[-1] + merge_row[-1]
            list_sheet_merge.append('{}:{}'.format(merge_start, merge_end))
    return sheet_cell_dict, list_sheet_merge


def iou(bbox1, bbox2):
    ### enhance text boxes
    x_i_top = max(bbox1[0] - 2, bbox2[0])
    y_i_top = max(bbox1[1] - 2, bbox2[1])
    x_i_bot = min(bbox1[2] + 2, bbox2[2])
    y_i_bot = min(bbox1[3] + 2, bbox2[3])
    if (x_i_bot - x_i_top) <= 0 or (y_i_bot - y_i_top) <= 0:
        return 0
        
    S_i = (x_i_bot - x_i_top)*(y_i_bot - y_i_top)
    S_bbox1 = (bbox1[2] - bbox1[0] + 4)*(bbox1[3]-bbox1[1] + 4)
    S_bbox2 = (bbox2[2] - bbox2[0])*(bbox2[3]-bbox2[1])
    
    return S_i/min(S_bbox1, S_bbox2)


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(min(boxAArea, boxBArea))
	# return the intersection over union value
	return iou