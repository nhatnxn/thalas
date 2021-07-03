from collections import OrderedDict
from swagger_server.inference.common import ReadBinaryImage, FastPipeline, Select, ToDict, ReturnUnion, MergeDict, \
    Identity
from swagger_server.inference.paddle_detection.pipeline import PaddlePipeline
from swagger_server.inference.corners.pipeline import CornersPipeline
from swagger_server.inference.vietocr.pipeline import VietocrPipeline


def create_pipeline_parameters():
    verbose = OrderedDict()
    p = FastPipeline([
        ReturnUnion([
            Identity(),
            FastPipeline([
                Select(keys="imageUrl"),
                ReadBinaryImage(verbose=verbose),
                ToDict(keys="image")])], joiner=MergeDict()),
        CornersPipeline(),
        PaddlePipeline(),
        VietocrPipeline()
    ])
    return p
