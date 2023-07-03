# dataset.py
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset

class GasDataset(Dataset):
    def __init__(self, file_path, start_year=None, end_year=None):
        # Load the data
        self.data = pd.read_csv(file_path)

        # Filter the data based on the year if start_year and end_year are provided
        if start_year is not None and end_year is not None:
            self.data = self.data[(self.data['Year'] >= start_year) & (self.data['Year'] <= end_year)]

        # TODO: Separate the features and labels
        features = ['Gangwondo', 'Seoul', 'Gyeonggido', 'Incheon', 'Gyeongsangnamdo', 'Gyeongsangbukdo', 'Gwangju', 'Daegu', 'Daejeon', 'Busan', 'Sejong', 'Ulsan', 'Jeollanamdo', 'Jeollabukdo', 'Jeju', 'Chungcheongnamdo', 'Chungcheongbukdo']
        
        k_gas_features = self.data[features].values
        k_gas_temp = self.data['Temperatrue'].values


        self.label = torch.tensor(k_gas_features).float()
        self.data = torch.tensor(k_gas_temp).float()

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return the features and label for the sample at the given index
        return self.x[idx], self.y[idx]
