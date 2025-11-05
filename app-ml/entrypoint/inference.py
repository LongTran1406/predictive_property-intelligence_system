"""
Inference Pipeline:
- Loads configuration
- Initializes production database
- Save predictions
"""

import os
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
sys.path.append(str(project_root / 'common'))
print(sys.path)

from common.utils import read_config
from common.data_manager import DataManager
from pipelines.pipeline_runner import PipelineRunner

if __name__ == '__main__':
    # Load config
    config = read_config(project_root / 'config' / 'config.yaml')

    # Initialize data manager
    data_manager = DataManager(config)

    # Load the dataset to run inference on
    # dataset_path = os.path.join(
    #     config['data_manager']['real_time_data_folder'],
    #     config['data_manager']['real_time_database_name']
    # ) 

    # df = data_manager.load_data(dataset_path)

    # Initialize Pipeline runner for inference
    pipeline_runner = PipelineRunner(config, data_manager)

    # Make inference based on the input
    y_pred = pipeline_runner.run_inference()
    print(y_pred)
    