import onnxruntime
from swagger_server.logger import create_logger


class CornersModel:
    def __init__(self):
        self.logger = create_logger(self.__class__.__name__)
        self.sess = None

    def load(self, cpkt):
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 4
        options.log_verbosity_level = 2
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = onnxruntime.InferenceSession(cpkt, sess_options=options)
        return self

    def transform(self, x):
        ort_inputs = {self.sess.get_inputs()[0].name: x}
        ort_outs = self.sess.run(None, ort_inputs)
        return ort_outs


def create_model():
    return CornersModel()
