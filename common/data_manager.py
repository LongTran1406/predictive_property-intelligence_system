import os 
import sys
from pathlib import Path

import pandas as pd
from typing import Dict, Any

class DataManager:
    """
    A utility class responsible for handling all data-related I/O operations
    and transformations used across the ML pipeline.

    Responsibilities:
    - Initializing production database
    - Loading and saving parquet files
    - Appending new data to existing datasets
    - Slicing or filtering data by timestamp
    - Saving predictions incrementally
    """

    def __init__(self, config: Dict[str, Any]):
        """
        DataManager class with a configuration dictionary.
        
        Args:
            config: Dict[str, Any]: Configuration params for path and filename.
        """
        self.config = config

    def initialize_prod_database(self) -> None:
        """
        Initialize the production database by copying raw databases
        into the production folder.

        Returns: None
        """
        # raw_data_path = os.path.join(self.config['data_manager']['raw_data_folder'],
        #                         self.config['data_manager']['raw_database_name'])
        
        # prod_data_path = os.path.join(self.config['data_manager']['prod_data_folder'],
        #                          self.config['data_manager']['prod_database_name'])
        
        # Read from the raw database
        # df = pd.read_csv(raw_data_path)

        # Save to the production database
        # df.to_parquet(prod_data_path, index = False)

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        """
        Load a DataFrame from a parquet file
        Args:
            path: str: path to the parquet file
        Returns:
            pd.DataFrame: Loaded data
        """
        return pd.read_parquet(path)
    