import pickle
import pandas as pd
import numpy as np

class ThalasDetector:

    def __init__(self, model_file):

        self.numerical_columns = ["SLHC","HST","HCT","MCV","MCH","MCHC","RDWCV","SLTC","SLBC","FERRITIN","FE"]

        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
    
    def read_data(self, csv_file):
        df = pd.read_csv(csv_file)
        
        minus_set = set(self.numerical_columns) - set(df.columns)

        if len(minus_set) != 0:
            return None
        
        return df[self.numerical_columns]


    def predict(self, csv_file):
        
        features = self.read_data(csv_file)

        if features is None:
            return -1
        
        preds = self.model.predict(features)  
        preds = np.where(preds>=0.6, 1, 0)

        return preds

if __name__ == '__main__':
    thalas_detector = ThalasDetector('model_fe.pkl')

    print(thalas_detector.predict('example.csv'))
    