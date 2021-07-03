import numpy as np

from swagger_server.inference.vietocr.vocab import Vocab
from torch.nn.functional import softmax
from swagger_server.inference.vietocr.utils import *
from swagger_server.inference.common import *
import pandas as pd
import onnxruntime


class OcrModel:
    def __init__(self, verbose: Union[dict, OrderedDict] = None):
        self.verbose = verbose
        self.sess = []
        self.vocab = Vocab()

    def load(self, cpkts):
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 4
        options.log_verbosity_level = 2
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        for cpkt in cpkts:
            self.sess.append(onnxruntime.InferenceSession(cpkt, sess_options=options))
        return self

    def predict(self, img, max_seq_length=128, sos_token=1, eos_token=2):
        src = img
        cnn_input = {self.sess[0].get_inputs()[0].name: src}
        src = self.sess[0].run(None, cnn_input)[0]
        src = torch.Tensor(src)
        src = src.transpose(-1, -2)
        src = src.flatten(2)
        src = src.permute(-1, 0, 1)
        src = to_numpy(src)
        encoder_input = {self.sess[1].get_inputs()[0].name: src}
        encoder_output, hidden = self.sess[1].run(None, encoder_input)
        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]
        max_length = 0
        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence)
            input_decoder = dict()
            for i, item in enumerate([to_numpy(tgt_inp)[-1], hidden, encoder_output]):
                input_decoder[self.sess[2].get_inputs()[i].name] = item
            output, hidden, last = self.sess[2].run(None,  input_decoder)
            output = torch.Tensor(np.expand_dims(output, axis=0))
            output = output.permute(1, 0, 2)
            output = softmax(output, dim=-1)
            output = output.to('cpu')
            values, indices = torch.topk(output, 5)
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)
            translated_sentence.append(indices)
            max_length += 1
            del output
        translated_sentence = np.asarray(translated_sentence).T
        return translated_sentence

    def predict_batch(self, images, set_bucket_thresh):
        batch_dict, indices = batch_process(images, set_bucket_thresh)
        list_keys = [i for i in batch_dict if batch_dict[i]
                     != batch_dict.default_factory()]
        result = list([])
        for width in list_keys:
            batch = batch_dict[width]
            batch = np.asarray(batch, dtype=np.float32)
            sent = self.predict(batch)
            batch_text = self.vocab.batch_decode(sent)
            result.extend(batch_text)
        z = zip(result, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        result, _ = zip(*sorted_result)

        return result

    def transform(self, data_inputs):
        img = data_inputs['image']
        cluster_box = data_inputs['cluster_box']
        first_col = sorted(cluster_box[0], key=lambda x: x[1])
        second_col = sorted(cluster_box[1], key=lambda x: x[1])
        cell_images = []
        for box in first_col:
            cell_images.append(img[box[1]:box[3], box[0]:box[2]])
        for box in second_col:
            cell_images.append(img[box[1]:box[3], box[0]:box[2]])
        texts = self.predict_batch(cell_images, set_bucket_thresh=40)
        start_idx_second_col = len(first_col)
        result = []
        for idx, left_cell in enumerate(first_col):
            right_text = find_right_text(start_idx_second_col, texts, left_cell, second_col)
            left_text = texts[idx]
            result.append([left_text, right_text])
        df = pd.DataFrame(result, columns=["TenXetNghiem", "KetQua"])
        first_row = 0
        for index, row in df.iterrows():
            if similar(row["TenXetNghiem"], "TÊN XÉT NGHIỆM"):
                first_row = index
                break
        df = df.iloc[first_row:]
        return df


def create_model():
    return OcrModel()