import pandas as pd
import numpy as np
from typing import Literal
    


class TimeSeriesDataset:
    def __init__(self, route: str, column:str, technique:Literal[None,'diff']):

        self.route = route
        self.column = column
    
    def load_dataset(self):
        """ this function reads the csv value and process it """
        
        csv_file = pd.read_csv(self.route)[self.column]

        if self.technique == 'diff':  csv_file = csv_file.diff()
        

        time_series = csv_file.to_numpy()
        return time_series
    

if __name__ == "__main__":
    None

