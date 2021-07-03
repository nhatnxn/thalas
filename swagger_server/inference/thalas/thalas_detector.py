import pickle
import numpy as np
import pandas as pd
from swagger_server.inference.common import get_checkpoint_path


class ThalasDetector:

    def __init__(self):
        self.features = ["SLHC", "HST", "HCT", "MCV", "MCH", "MCHC", "RDWCV", "SLTC", "SLBC", "FERRITIN", "FE", "HBA1", "HBA2", "MODE"]
        self.class_names = {0: "nothalas", 1: "thalas", 2: 'misc'}
        self.models = []
        model_names = ["model_9params.pkl", 'model_11params.pkl', 'model_13params.pkl']
        for name in model_names:
            with open(get_checkpoint_path('thalas', name), 'rb') as f:
                self.models.append(pickle.load(f))

    def predict_9param(self, features):
        inputs = dict()
        if len(features) != len(self.features):
            return self.class_names[2]
        for i, key in enumerate(self.features):
            if i > 8:
                break
            inputs[key] = [features.get(key)]
        features = pd.DataFrame(data=inputs)
        preds = self.models[0].predict(features)
        return self.class_names[preds[0]]

    def predict_11param(self, features):
        inputs = dict()
        if len(features) != len(self.features):
            return self.class_names[2]
        for i, key in enumerate(self.features):
            if i > 10:
                break
            inputs[key] = [features.get(key)]
        features = pd.DataFrame(data=inputs)
        preds = self.models[1].predict(features)
        preds = np.where(preds >= 0.6, 1, 0)
        return self.class_names[preds[0]]

    def predict_13param(self, features):
        inputs = dict()
        if len(features) != len(self.features):
            return self.class_names[2]
        for i, key in enumerate(self.features):
            if key == "MODE":
                break
            inputs[key] = [features.get(key)]
        features = pd.DataFrame(data=inputs)
        preds = self.models[2].predict(features)
        return self.class_names[preds[0]]
