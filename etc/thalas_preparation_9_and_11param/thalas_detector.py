import pickle
import pandas as pd
import numpy as np

class ThalasDetector:

    def __init__(self, model_path, pred_option):
        
        if pred_option == "9_features":
            self.numerical_columns = ["HST", "MCV", "MCH", "MCHC", "RDWCV", "SLTC", "SLBC", "SLHC", "HCT"]
        elif pred_option == "13_features":
            self.numerical_columns = ['SLHC', 'HST', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDWCV', 'SLTC', 'SLBC',
                                        'FE', 'FERRITIN', 'HBA1', 'HBA2']

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def read_data(self, csv_file):
        df = pd.read_csv(csv_file).iloc[0]

        condition = all([item in df.columns.to_list() for item in self.numerical_columns])
        if condition == False:
            return None
        
        return df[self.numerical_columns]


    def predict(self, csv_file):
        
        features = self.read_data(csv_file)

        if features is None:
            return -1
        
        preds = self.model.predict(features)  
        # preds = np.where(preds>=0.6, 1, 0)

        return preds

if __name__ == '__main__':

    ## Test on model 9 features
    pred_option = "9_features"
    model_path = "model_9params.pkl"

    ## Test on model 13 features
    #pred_option = "13_features"
    #model_path = "model_13_features.pkl"

    thalas_detector = ThalasDetector(model_path, pred_option)

    print(thalas_detector.predict('example.csv'))
    
