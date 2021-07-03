from paddleocr import PaddleOCR
from swagger_server.logger import create_logger


class CraftModel:
    def __init__(self):
        self.logger = create_logger(self.__class__.__name__)
        self.sess = None

    def load(self):
        self.sess = PaddleOCR(lang="latin")
        return self

    def transform(self, x):
        return self.sess.ocr(x, rec=False)


def create_model():
    return CraftModel()
