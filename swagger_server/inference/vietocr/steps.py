from neuraxle.base import NonFittableMixin, ExecutionContext, DataContainer
from collections import OrderedDict
from swagger_server.inference.common import FastStep
from swagger_server.logger import create_logger
from typing import Dict
import numpy as np
import time
from swagger_server.inference.vietocr.utils import remove_small_box, cluster_bounding_box, remove_small_cluster


class VietocrPreprocessStep(NonFittableMixin, FastStep):
    RESIZE_SIZE = 640

    def __init__(self, verbose: [dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.verbose = verbose

    def preprocess(self, box_texts):
        boxes = np.array(box_texts, dtype=int)
        top_lefts = np.stack([boxes[:, :, 0].min(axis=1), boxes[:, :, 1].min(axis=1)], axis=1)
        bot_rights = np.stack([boxes[:, :, 0].max(axis=1), boxes[:, :, 1].max(axis=1)], axis=1)
        boxes = np.concatenate([top_lefts, bot_rights], axis=1).tolist()
        boxes = sorted(boxes)
        boxes = remove_small_box(boxes)
        cluster_box = dict(enumerate(cluster_bounding_box(boxes), 1))
        cluster_box = remove_small_cluster(cluster_box)
        return cluster_box

    def transform(self, data_inputs: [Dict, np.array]):
        if isinstance(data_inputs, str):
            return data_inputs
        data_inputs['cluster_box'] = self.preprocess(data_inputs['PaddleDection'])
        return data_inputs


class VietocrProcessStep(NonFittableMixin, FastStep):
    def __init__(self, create_model, checkpoint, verbose: [dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.create_model = create_model
        self.checkpoint = checkpoint
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

        self.model = self.create_model().load(self.checkpoint)

        self.logger.info("Load {cpkt} checkpoint successfully"
                         .format(cpkt=self.checkpoint))
        self.is_initialized = True

        return self

    def transform(self, data_inputs):
        if self.verbose is not None:
            self.verbose[self.name + "_Time"] = time.time()
        if isinstance(data_inputs, str):
            return data_inputs
        pred = self.model.transform(data_inputs)
        return pred

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        data_inputs = data_container.data_inputs
        output = self.transform(data_inputs)
        data_container.set_data_inputs(output)
        return data_container


class VietocrPostprocessStep(NonFittableMixin, FastStep):
    def __init__(self, verbose: [dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.verbose = verbose

    def transform(self, data_inputs):
        return data_inputs
