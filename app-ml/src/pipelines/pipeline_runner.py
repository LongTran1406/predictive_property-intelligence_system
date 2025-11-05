import os
import sys
from pathlib import Path
import glob
import os
import pandas as pd
from datetime import datetime

# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()


project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
sys.path.append(str(project_root / 'app-etl'))

from typing import Dict, Any
from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline
from pipelines.postprocessing import PostProcessingPipeline
from pipelines.inference import InferencePipeline
# from tasks.transform import TransformTool
from tasks.extract import ExtractTool

class PipelineRunner:
    """
    A class that orchestrates the execution of all stages in the ML pipeline

    This includes:
    - Preprocessing
    - Feature engineering
    - Training
    - Inference
    - Postprocessing
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initialize the pipeline runner and its pipeline components.

        Args:
            config (Dict[str, Any]): Dictionary containing all pipeline configurations
            data_manager (DataManager): An DataManager instance for managing I/O operations on data.
        """

        self.config = config
        self.data_manager = data_manager

        # Initialize individual pipeline components
        
        self.raw_folder = Path(self.config['data_manager']['raw_data_folder'])
        self.real_time_folder = Path(self.config['data_manager']['real_time_data_folder'])
        self.real_time_file = Path(self.real_time_folder / self.config['data_manager']['real_time_database_name'])

        # self.transform_pipeline = TransformTool(self.config, raw_file = self.raw_file, checkpoint_file = self.checkpoint_file)
        self.extract_pipeline = ExtractTool(self.config)
        self.preprocessing_pipeline = PreprocessingPipeline(self.config)
        self.feature_engineering_pipeline = FeatureEngineeringPipeline(self.config)
        self.training_pipeline = TrainingPipeline(self.config)
        self.postprocessing_pipeline = PostProcessingPipeline(self.config)
        self.inference_pipeline = InferencePipeline(self.config)

        self.api_key = os.getenv("LOCATIONIQ_KEY_4")
    
    def run_trainning(self) -> None:
        """
        Run the full training pipeline:
        1. Load and preprocess data
        2. Perform feature engineering
        3. Train the model
        4. Save the trained model

        Returns:
            None
        """
        now = datetime.now()
        date_str = now.strftime('%Y%m%d')

        # self.raw_data_path = (
        #     Path(self.config['data_manager']['raw_data_folder'])
        #     / f"year={now.year}"
        #     / f"month={now.strftime('%m')}"
        #     / f"day={now.strftime('%d')}"
        #     / f"database_{date_str}.parquet"
        # )

        self.prod_data_path = (
            Path(self.config['data_manager']['prod_data_folder'])
            / f"year={now.year}"
            / f"month={now.strftime('%m')}"
            / f"day={now.strftime('%d')}"
            / f"database_cleaned_{date_str}.parquet"
        )

        self.prod_data_path_csv = (
            Path(self.config['data_manager']['prod_data_folder'])
            / f"year={now.year}"
            / f"month={now.strftime('%m')}"
            / f"day={now.strftime('%d')}"
            / f"database_cleaned_{date_str}.csv"
        )
        
        # print(self.prod_data_path)
        # print(self.raw_folder)
        # print(os.listdir(self.raw_folder))
        all_files_raw = glob.glob(os.path.join(self.raw_folder, "**", "*.parquet"), recursive=True)
        print(all_files_raw)
        df = pd.concat([pd.read_parquet(f) for f in all_files_raw], ignore_index=True)
        df = df.drop(columns=['id'])
        print(df.size)
        df = self.preprocessing_pipeline.run(df)
        os.makedirs(os.path.dirname(self.prod_data_path), exist_ok=True)
        df.to_parquet(self.prod_data_path)
        df.to_csv(self.prod_data_path_csv)


        folder_path = self.config['data_manager']['prod_data_folder']
        print(folder_path)
        all_files = glob.glob(os.path.join(folder_path, "**", "*.parquet"), recursive=True)
        print(all_files)

        # read and concat
        df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
        
        df = self.feature_engineering_pipeline.run(df)
        
        path = Path(self.config['data_manager']['ml_ready_folder']) / Path(self.config['data_manager']['ml_database_name'])
        df.to_parquet(path, index=False)

        model = self.training_pipeline.run(df)
        self.postprocessing_pipeline.run_train(model)
        return

    def run_inference(self) -> None:
        """
        Run the full inference pipeline:
        1. Preprocess, transform, and predict
        2. Postprocess and store the prediction
        Args:
            None
        Returns:
            None
        """

        # Step 1
        df = pd.read_parquet(self.real_time_file)
        lat, lon, postcode, city = self.extract_pipeline.extract_address(df['address'], self.api_key)
        df['lat'], df['lon'], df['postcode'], df['city'] = lat, lon, postcode, city 
        # df = self.transform_pipeline.transform(df)
        df = df[
            ['bathroom_nums', 'bedroom_nums', 'car_spaces', 'land_size',
            'lat', 'lon', 'postcode', 'city']
        ]

        df = self.preprocessing_pipeline.run(df)
        df = self.feature_engineering_pipeline.run(df)
        
        y_pred = self.inference_pipeline.run(x = df)
        
        # Step 2:
        y_pred = self.postprocessing_pipeline.run_inference(y_pred)
        return y_pred, df