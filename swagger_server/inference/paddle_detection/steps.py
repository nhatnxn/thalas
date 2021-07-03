from neuraxle.base import NonFittableMixin, ExecutionContext, DataContainer
from collections import OrderedDict
from swagger_server.inference.common import FastStep
from swagger_server.logger import create_logger
from swagger_server.inference.paddle_detection.utils import resize_image, get_dewarped_table
from typing import Dict
import numpy as np
import time
import cv2


class PaddlePreprocessStep(NonFittableMixin, FastStep):

    def __init__(self, verbose: [dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.verbose = verbose

    def transform(self, data_inputs: [Dict, np.array]):
        if self.verbose is not None:
            self.verbose[self.name + "_Time"] = time.time()
        if isinstance(data_inputs, Dict):
            image = data_inputs["image"]
        else:
            image = data_inputs
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corners = data_inputs['corners']
        if not corners:
            return "BADIMAGE"
        image = get_dewarped_table(image, corners)
        cv2.imwrite("/home/thanh/Downloads/Thalas%20OCR/samples/test.png", image)
        image = resize_image(image)
        data_inputs['image'] = image
        return data_inputs


class PaddleProcessStep(NonFittableMixin, FastStep):
    def __init__(self, create_model, verbose: [dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.create_model = create_model
        self.verbose = verbose
        self.logger = create_logger(self.__class__.__name__)

    def setup(self) -> 'SkinDetectionProcessStep':
        """
        Setup model graph
        :return: step
        :rtype: SkinDetectionProcessStep
        """
        if self.is_initialized:
            return self
        self.model = self.create_model().load()
        self.is_initialized = True

        return self

    def transform(self, data_inputs):
        if self.verbose is not None:
            self.verbose[self.name + "_Time"] = time.time()
        if isinstance(data_inputs, str):
            return data_inputs
        pred = self.model.transform(data_inputs['image'])
        return data_inputs, pred

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        data_inputs = data_container.data_inputs
        output = self.transform(data_inputs)
        data_container.set_data_inputs(output)
        return data_container


class PaddlePostprocessStep(NonFittableMixin, FastStep):

    def __init__(self, verbose: [dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.verbose = verbose

    def transform(self, data_inputs):
        if isinstance(data_inputs, str):
            return data_inputs
        data_inputs, pred = data_inputs
        data_inputs['PaddleDection'] = pred
        return data_inputs
