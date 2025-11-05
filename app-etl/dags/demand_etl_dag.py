from pathlib import Path
import sys
from datetime import datetime
import os
from typing import Dict, Any
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))
sys.path.append(str(project_root / 'app-etl'))

os.chdir(project_root) # Change directory to read the files from ./data folder

from tasks.extract import ExtractTool
from tasks.transform import TransformTool
from tasks.load import LoadTool
from common.utils import read_config

class HousePriceETL:
    """
    A complete ETL Pipeline for handling the extract, transform and process steps
    Typical workflow:
    1. Extract (extract.py): Extract raw dataset from by scraping Realestate website
    2. Transform (transform.py): Transform dataset (preprocessing) and serve as Silver data
    3. Load (load.py): Load cleaned dataset to the datalake 
    """

    def __init__(self, config: Dict[str, Any]):
        """
        HousePriceETL class with a configuration dictionary for setting up ExtractTool, TransformTool, and LoadTool.
        
        Args:
            config: Dict[str, Any]: Configuration params for path and filename.
        Returns:
            None
        """
        self.config = config

        now = datetime.now()
        self.raw_folder = Path(self.config['data_manager']['raw_data_folder']) / f"year={now.year}" / f"month={now.strftime('%m')}" / f"day={now.strftime('%d')}"
        self.raw_file = self.raw_folder / f"{self.config['data_manager']['raw_database_name'].replace('.parquet','')}_{now.strftime('%Y%m%d')}.parquet"
        # print(self.raw_file)
        # self.checkpoint_file = self.raw_folder / "geocode_partial.csv"
        self.extract_tool = ExtractTool(self.config)
        # self.transform_tool = TransformTool(self.config, self.raw_file, self.checkpoint_file)
        self.load_tool = LoadTool(self.config)

    @task
    def extract_task(self):
        df = self.extract_tool.extract_property()
        print("Extract complete")
        return None
    
    @task
    def load_task(self, df):
        self.load_tool.load(df)
        print("Load complete")
        return None

    @flow(task_runner=ConcurrentTaskRunner())
    def house_price_etl_flow(self):
        # Task chaining
        self.extract_task()
        # self.load_task(raw_data)

config_path = project_root / 'config' / 'config.yaml'
config = read_config(config_path)
etl = HousePriceETL(config)
etl.house_price_etl_flow()