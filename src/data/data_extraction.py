import pandas as pd
import numpy as np
from typing import Literal
    


class TimeSeriesDataset:
    def __init__(self, route: str, column:str, technique:Literal[None,'diff'], lookback:int = 0 ):

        self.route = route
        self.column = column
        self.lookback = lookback
        self.technique = technique
    
    def load_dataset(self):
        """ this function reads the csv value and process it """
        
        csv_file = pd.read_csv(self.route)[self.column]

        if self.technique == 'diff':  csv_file = csv_file.diff().dropna()

        time_series = csv_file.to_numpy()
        
        if self.lookback: return self._create_sequence(time_series, self.lookback)
        
        
        return time_series
    

    def _create_sequence(self, data, lookback):
        X, y = [], []

        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])

        return np.array(X), np.array(y)
        

if __name__ == "__main__":
    None

