from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples
from swagger_server.inference.common import get_checkpoint_path, FastPipeline
from typing import Union
from collections import OrderedDict
from .steps import VietocrPreprocessStep, VietocrProcessStep, VietocrPostprocessStep
from .model import create_model


class VietocrPipeline(FastPipeline):
    def __init__(self, verbose: Union[dict, OrderedDict]=None):
        cpkt = [get_checkpoint_path('ocr', 'model_cnn.onnx'), get_checkpoint_path('ocr', 'model_encoder.onnx'),
                get_checkpoint_path('ocr', 'model_decoder.onnx')]
        self.verbose = verbose
        steps = [
            VietocrPreprocessStep(),
            VietocrProcessStep(
                checkpoint=cpkt,
                create_model=create_model
            ).setup(),
            VietocrPostprocessStep()
        ]

        for step in steps:
            step.verbose = self.verbose
        FastPipeline.__init__(self, steps)

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]) -> BaseStep:
        for name, step in self.steps_as_tuple:
            step.set_hyperparams(hyperparams)
        return FastPipeline.set_hyperparams(self, hyperparams)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        After loading the last checkpoint, transform each pipeline steps
        :param data_container: the data container to transform
        :return: transformed data container
        """
        steps_left_to_do, data_container = self._load_checkpoint(data_container, context)

        for step_name, step in steps_left_to_do:
            data_container = step.handle_transform(data_container, context)

        return data_container