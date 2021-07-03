import time
import models
import torch
import onnxruntime
import numpy as np
from utils.activation import SiLU
from models.experimental import attempt_load


class Exporter:

    def __init__(self, model_path):
        self.model = attempt_load(model_path, map_location='cpu')  # load FP32 model
        self.model.eval()
        self.model.onnx_dynamic = False

    def export(self):
        for k, m in self.model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.7.0 compatability
            if isinstance(m, models.common.Conv):
                m.act = SiLU()
        x = torch.randn([1, 3, 640, 640])

        torch.onnx.export(self.model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          'model.onnx',  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],
                          dynamic_axes={'input': {2: 'height', 3: 'width'}})

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def test(self):
        torch_time_avg = 0
        onnx_time_avg = 0
        n = 10
        ort_session = onnxruntime.InferenceSession("model.onnx")
        for i in range(n):
            x = torch.randn(1, 3, 640, 640)

            torch_t = time.time()
            torch_out = self.model(x)
            torch_out = torch_out[0][0].detach().cpu().numpy()

            torch_t = time.time() - torch_t

            onnx_t = time.time()
            ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(x)}
            ort_outs = ort_session.run(None, ort_inputs)[0]
            # print(ort_session.run(None, ort_inputs))
            onnx_t = time.time() - onnx_t
            np.testing.assert_allclose(torch_out, ort_outs, rtol=1e-03, atol=1e-03)
            torch_time_avg += torch_t
            onnx_time_avg += onnx_t
            print("Smart Beauty %d: Pytorch took time:%0.3fs - OnnxRuntime took time: %0.3fs" % (
                i, torch_t, onnx_t))
        print("Torch avg time:%0.3f - OnnxRuntime avg time: %0.3f" % (torch_time_avg, onnx_time_avg))


if __name__ == "__main__":
    exporter = Exporter("./weights/yolov5s_thalas.pt")
    exporter.export()
    exporter.test()
