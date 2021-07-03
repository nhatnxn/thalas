from swagger_server.logger import create_logger
from werkzeug.exceptions import BadRequest
from collections import OrderedDict
from neuraxle.base import NonFittableMixin, BaseStep, ExecutionContext, ExecutionMode, Identity
from neuraxle.union import FeatureUnion
from neuraxle.pipeline import Pipeline
from neuraxle.data_container import DataContainer
from requests.exceptions import HTTPError
from typing import List, Any, Union, Dict
from PIL import Image
import requests
import numpy as np
import os
import re
import time
import cv2
import io

INFERENCE_PATH = os.path.dirname(__file__)
WEIGTHS_PATH = os.path.join(os.path.dirname(os.path.dirname(INFERENCE_PATH)), "weights")


def get_checkpoint_path(name, filename):
    return os.path.join(WEIGTHS_PATH, name, filename)


class FastDataContainer(DataContainer):
    def hash_summary(self):
        return "hash_summary"


class FastStep(BaseStep):
    """
    In the origin BaseStep, the method hash_data_container very slow, so return a uuid will speed up
    """

    def hash(self, data_container):
        return "hash"

    def summary_hash(self, data_container: DataContainer) -> str:
        return "summary_hash"


class FastPipeline(Pipeline):
    """
    In the origin BaseStep, the method hash_data_container very slow, so return a uuid will speed up
    """

    def transform(self, data_inputs: Any):
        """
        After loading the last checkpoint, transform each pipeline steps

        :param data_inputs: the data input to transform
        :return: transformed data inputs
        """
        data_container = FastDataContainer(data_inputs=data_inputs, current_ids="0", expected_outputs=[])
        data_container = self.hash_data_container(data_container)

        context = ExecutionContext(root=self.cache_folder, execution_mode=ExecutionMode.TRANSFORM)
        context = context.push(self)

        data_container = self._transform_data_container(data_container, context)

        return data_container.data_inputs

    def hash(self, data_container):
        return "hash"

    def summary_hash(self, data_container: DataContainer) -> str:
        return "summary_hash"


class ReturnUnion(FeatureUnion):
    def __init__(self, *args, **kwargs):
        FeatureUnion.__init__(self, *args, **kwargs)

    def _transform_data_container(self, data_container, context):
        """
        :param data_container: data container
        :param context: execution context
        :return: the transformed data_inputs.
        """
        # Don't use joblib parallel it make process very slow
        data_containers = [
            step.handle_transform(data_container.copy(), context)
            for _, step in self.steps_as_tuple[:-1]
        ]

        return DataContainer(
            data_inputs=data_containers,
            current_ids=data_container.current_ids,
            summary_id=data_container.summary_id,
            expected_outputs=data_container.expected_outputs,
            sub_data_containers=data_container.sub_data_containers
        )


class Parallel(FeatureUnion):
    def __init__(self, *args, **kwargs):
        FeatureUnion.__init__(self, *args, **kwargs)

    def _transform_data_container(self, data_container, context):
        """
        :param data_container: data container
        :param context: execution context
        :return: the transformed data_inputs.
        """
        # Don't use joblib parallel it make process very slow
        data_containers = [
            step.handle_transform(data_container.copy(), context)
            for _, step in self.steps_as_tuple[:-1]
        ]

        return DataContainer(
            data_inputs=data_containers,
            current_ids=data_container.current_ids,
            summary_id=data_container.summary_id,
            expected_outputs=data_container.expected_outputs,
            sub_data_containers=data_container.sub_data_containers
        )


class MergeDict(NonFittableMixin, FastStep):
    def __init__(self):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)

    def _transform_data_container(self, data_container, context):
        """
        Handle transform.
        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = self.transform([dc.data_inputs for dc in data_container.data_inputs])
        data_container = DataContainer(data_inputs=data_inputs, current_ids=data_container.current_ids,
                                       expected_outputs=data_container.expected_outputs)
        data_container.set_data_inputs(data_inputs)

        return data_container

    def transform(self, data_inputs: List[Dict]):
        new_dict = {}
        for d in data_inputs:
            new_dict.update(d)

        return new_dict


class ToDict(NonFittableMixin, FastStep):
    def __init__(self, keys: Union[List[str], str]):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.keys = keys

    def _transform_data_container(self, data_container, context):
        """
        Handle transform.
        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = data_container.data_inputs
        if isinstance(self.keys, list) and len(self.keys) > 1:
            if not isinstance(data_inputs, list) and not isinstance(data_inputs, tuple):
                data_inputs = [data_inputs]
        data_inputs = self.transform(data_inputs)
        data_container.set_data_inputs(data_inputs)

        return data_container

    def transform(self, data_inputs: List[Any]):
        if isinstance(self.keys, list) and len(self.keys) > 1:
            return dict(zip(self.keys, data_inputs))
        return {self.keys:data_inputs}


class Select(NonFittableMixin, FastStep):
    def __init__(self, keys: Union[List[str], str]):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)

        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys

    def _transform_data_container(self, data_container, context):
        """
        Handle transform.
        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = data_container.data_inputs
        if not isinstance(data_inputs, dict):
            raise ValueError("data_inputs for Select step must be dict")
        data_inputs = self.transform(data_inputs)
        data_container.set_data_inputs(data_inputs)

        return data_container

    def transform(self, data_inputs: dict):
        result = []
        for k in self.keys:
            result.append(data_inputs.get(k))
        if len(result) == 1:
            result = result[0]
        return result


class ReadBinaryImage(NonFittableMixin, FastStep):
    def __init__(self, verbose: Union[dict, OrderedDict] = None):
        FastStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.logger = create_logger(self.name + "_Time")
        self.verbose = verbose
        self.url_pattern = r"^https?:\/\/.+"

    def transform(self, data_inputs):
        try:
            if isinstance(data_inputs, str):
                if self.verbose is not None:
                    self.verbose[self.name + "_Time"] = time.time()
                if re.match(self.url_pattern, data_inputs):
                    r = requests.get(data_inputs, stream=True)
                    r.raise_for_status()
                    buff = r.content
                else:
                    # url is local file for testing
                    with open(data_inputs, "rb") as fobj:
                        buff = fobj.read()
                if self.verbose is not None:
                    self.verbose[self.name + "_Time"] = time.time() - self.verbose[self.name + "_Time"]
                img = np.frombuffer(buff, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            else:
                img = Image.open(io.BytesIO(data_inputs.stream.read())).convert('RGB')
                if img.mode.lower() == "rgba":
                    img = Image.merge("RGB", img.split()[:3])
                self.logger.info(f'Read image from {data_inputs}')
                return np.asarray(img)
        except HTTPError as http_err:
            self.logger.info(f'HTTP error occurred: {http_err}')
            raise BadRequest(f"Cannot read image from url {data_inputs}")
        except Exception as err:
            self.logger.info(f'Other error occurred: {err}')
            raise BadRequest(f"Cannot read image from url {data_inputs}")