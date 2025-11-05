import pandas as pd
from typing import Dict, Any
import numpy as np
import json
import os

class FeatureEngineeringPipeline:
    """
    A pipeline for creating and engineering features from preprocessed data.

    This class handles feature engineering steps:
    - Creating some  features for better eda and model training
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Pipeline with a configuration dictionary.
        
        Args:
            config: Dict[str, Any]: Configuration params for path and filename.
        Returns:
            None
        """
        self.config = config
        self.mapping_path = os.path.join(
            self.config['feature_engineering']['mapping_path'],
            self.config['feature_engineering']['mapping_file']
        )
        # Coordinates of each city for calculating distance later
        self.city_coords = {
            0: (-33.8688, 151.2093),  # Sydney
            1: (-32.9267, 151.7789),  # Newcastle
            2: (-34.4278, 150.8931)   # Wollongong
        }
    
    def price_per_m2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to create price_per_m2 feature

        Args:
            pd.DataFrame: Input dataframe for processed
        Returns:
            pd.DataFrame: Output dataframe after creating feature
        """
        if self.config['features']['target_column'] in df.columns:
            # df['price_per_m2'] = df['price'] / df['land_size']
            df[self.config['features']['price_per_m2']] = (
                df[self.config['features']['target_column']] /
                df[self.config['features']['land_size']]
            )
        return df
    
    def add_distance_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to create add_distance_feature

        Args:
            pd.DataFrame: Input dataframe for processed
        Returns:
            pd.DataFrame: Output dataframe after creating feature
        """
        df = df.copy()
        features = self.config['features']

        # Map city coordinates
        lat_c = df[features['city']].map(lambda c: self.city_coords[c][0])
        lon_c = df[features['city']].map(lambda c: self.city_coords[c][1])
        # lat_c = df['city'].map(lambda c: self.city_coords[c][0])
        # lon_c = df['city'].map(lambda c: self.city_coords[c][1])

        # Euclidean distance in degrees
        df[features['dist_to_city']] = np.sqrt(
            (df[features['lat']] - lat_c) ** 2 + (df[features['lon']] - lon_c) ** 2
        ) * 1000
        # df['dist_to_city'] = np.sqrt((df['lat'] - lat_c) ** 2 + (df['lon'] - lon_c) ** 2) * 1000

        df[features['city']] = df[features['city']].apply(
            lambda x: 1 if x == 'Newcastle' else (2 if x == 'Wollongong' else 0)
        )
        # df['city'] = df['city'].apply(
        #     lambda x: 1 if x == 'Newcastle' else (2 if x == 'Wollongong' else 0)
        # )
        return df

    def avg_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to create avg_info feature, including postcode_avg_price and postcode_avg_price_per_m2

        Args:
            pd.DataFrame: Input dataframe for processed
        Returns:
            pd.DataFrame: Output dataframe after creating feature
        """
        target_col = self.config['features']['target_column']
        postcode_col = self.config['features']['postcode']
        price_per_m2_col = self.config['features']['price_per_m2']

        avg_price_col = self.config['features']['avg_price_by_postcode']
        avg_price_m2_col = self.config['features']['postcode_avg_price_per_m2']

        if target_col in df.columns:
            postcode_avg_price = df.groupby(postcode_col)[target_col].mean().to_dict()
            df[avg_price_col] = df[postcode_col].map(postcode_avg_price)
            # df['avg_price_by_postcode'] = df['postcode'].map(postcode_avg_price)

            postcode_avg_price_per_m2 = df.groupby(postcode_col)[price_per_m2_col].mean().to_dict()
            df[avg_price_m2_col] = df[postcode_col].map(postcode_avg_price_per_m2)
            # df['postcode_avg_price_per_m2'] = df['postcode'].map(postcode_avg_price_per_m2)

            print(self.mapping_path)
            os.makedirs(os.path.dirname(self.mapping_path), exist_ok=True)

            with open(self.mapping_path, "w") as f:
                json.dump({
                    "avg_price": postcode_avg_price,
                    "avg_price_per_m2": postcode_avg_price_per_m2
                }, f)
                
        else:
            # print(self.mapping_path)
            # print(df['postcode'])
            with open(self.mapping_path, "r") as f:
                data = json.load(f)
                self.postcode_avg_price = data["avg_price"]
                self.postcode_avg_price_per_m2 = data["avg_price_per_m2"]

            df[avg_price_col] = df[postcode_col].astype(str).map(self.postcode_avg_price)
            df[avg_price_m2_col] = df[postcode_col].astype(str).map(self.postcode_avg_price_per_m2)
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline on the input DataFrame
        Args:
            df: pd.DataFrame: Input DataFrame
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """

        df = self.price_per_m2(df)
        df = self.add_distance_feature(df)
        df = self.avg_info(df)
        return df
