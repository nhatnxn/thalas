import sys
import os
import cv2
from paddleocr import PaddleOCR
import json
import warnings
import os

os.environ['ONNX_PATH'] = os.getcwd()
warnings.filterwarnings("ignore")

sys.path.insert(0, 'models/vietocr')
from utils import ocr_table, utils

if __name__ == '__main__':

    # load model
    paddleocr = PaddleOCR(lang="latin")
    vietocr = utils.load_model_OCR()

    # read an image 
    test_path = "/home/thanh/thalas-ai-service/sample"
    images_test = [img for img in os.listdir(test_path) if img.endswith((".jpg", ".png", ".tif"))]
    
    for image_name in images_test[1:3]:
        image = cv2.imread(f"{test_path}/{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # vietocr table
        out = ocr_table.ocr_table(paddleocr, vietocr, image, image_name)  
        print(json.loads(out))      
