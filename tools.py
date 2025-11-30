# import dependencies
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# cvs formatted dataset class
class earthQuakeData(Dataset):
    def __init__(self, file_path, feature_columns, target_column):
        """
        * file_path (string): a path to csv file
        * feature_columns (string): column name(s) in the csv file to include as feature(s) in this dataset
        * target_column (string): a column name that will be used as the label(ground truth) in this dataset
        """
        # reading csv with pandas: https://www.w3schools.com/python/pandas/ref_df_iloc.asp
        data_frame = pd.read_csv(file_path)
        self.X = data_frame[feature_columns].values.astype(np.float32)

        # print out how current features in this dataset looks
        print(f"Current selected features: {feature_columns}")
        print(f"Current feature shape: {self.X.shape}")

        # in the case that column name for the ground truth label is 'sig'
        if target_column == 'sig':
            self.y = data_frame[target_column].values
            print(f"Current label: '{target_column}', and its first 5 values: {self.y[:5]}")

        # in the case that column name for the ground truth label is 'alert'
        elif target_column == 'alert':
            # encoding label with sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
            le = LabelEncoder()
            self.y = le.fit_transform(data_frame[target_column])
            encoding_map = {}
            for i, label in enumerate(le.classes_):
                encoding_map[label] = i
            
            # print out how current labels in this dataset looks
            print(f"Current labels: {encoding_map}")

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = torch.tensor(self.X[index])
        y = torch.tensor(self.y[index])
        return x, y
