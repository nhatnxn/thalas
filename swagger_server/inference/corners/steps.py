import torch
from neuraxle.base import NonFittableMixin, ExecutionContext, DataContainer
from collections import OrderedDict
from swagger_server.inference.common import FastStep
from swagger_server.logger import create_logger
from swagger_server.inference.corners.utils import letterbox, non_max_suppression, output_to_target, scale_coords, \
    polygon_from_corners, increase_border
from typing import Dict
import numpy as np
import time


class CornersPreprocessStep(NonFittableMixin, FastStep):
    RESIZE_SIZE = 640

    def __init__(self, verbose: [dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.verbose = verbose

    def preprocess(self, image, stride=32):
        img = letterbox(image, self.RESIZE_SIZE, stride=stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def transform(self, data_inputs: [Dict, np.array]):
        if self.verbose is not None:
            self.verbose[self.name + "_Time"] = time.time()
        if 'corners' in data_inputs.keys():
            return data_inputs
        if isinstance(data_inputs, Dict):
            image = data_inputs["image"]
        else:
            image = data_inputs
        w, h = image.shape[:2]
        resized_image = self.preprocess(image)
        w_r, h_r = resized_image.shape[2:]
        return resized_image, (w, w_r), (h, h_r), data_inputs


class CornersProcessStep(NonFittableMixin, FastStep):
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
        if isinstance(data_inputs, Dict):
            return data_inputs
        resized_image, w, h, data_inputs = data_inputs
        if 'corners' in data_inputs.keys():
            return data_inputs
        if self.verbose is not None:
            self.verbose[self.name + "_Time"] = time.time()
        pred = self.model.transform(resized_image)
        return pred, w, h, data_inputs

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        data_inputs = data_container.data_inputs
        output = self.transform(data_inputs)
        data_container.set_data_inputs(output)
        return data_container


class CornersPostprocessStep(NonFittableMixin, FastStep):
    CONF_THES = 0.3
    IOU_THRES = .5
    PADDING_SIZE = 8
    NO = 9

    def __init__(self, verbose: [dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.verbose = verbose

    def refine_boxes(self, ort_outs):
        z = []
        for i in range(int(len(ort_outs) / 6)):
            preds = ort_outs[i * 6:(i + 1) * 6]
            for k, pred in enumerate(preds):
                preds[k] = torch.tensor(pred)
            preds[0][..., 0:2] = (preds[0][..., 0:2] * 2. - 0.5 + preds[1]) * preds[3]  # xy
            preds[0][..., 2:4] = (preds[0][..., 2:4] * 2) ** 2 * preds[4]  # wh
            z.append(preds[0].view(preds[5], -1, self.NO))
        ort_outs = torch.cat(z, 1).numpy()
        return ort_outs

    def transform(self, data_inputs):
        if isinstance(data_inputs, Dict):
            return data_inputs
        pred, w, h, data_inputs = data_inputs
        if 'corners' in data_inputs.keys():
            return data_inputs
        pred = self.refine_boxes(pred)
        out = non_max_suppression(torch.Tensor(pred), conf_thres=self.CONF_THES, iou_thres=self.IOU_THRES,
                                  multi_label=True)
        target = output_to_target(out)
        target = torch.from_numpy(target)
        target[:, 2:6] = scale_coords((w[1], h[1]), target[:, 2:6], (w[0], h[0])).round()
        target = target.numpy()
        if target is None:
            data_inputs['corners'] = []
            return data_inputs
        pts = polygon_from_corners(target)
        if pts is None:
            data_inputs['corners'] = []
            return data_inputs
        pts = pts.astype(int)
        if pts is None:
            data_inputs['corners'] = []
            return data_inputs
        else:
            corners = increase_border(pts, self.PADDING_SIZE)
            corners = [(int(p[0]), int(p[1])) for p in corners]
            data_inputs['corners'] = corners
            return data_inputs
