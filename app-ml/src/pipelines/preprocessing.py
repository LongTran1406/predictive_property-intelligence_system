import pandas as pd
from typing import Dict
import numpy as np

class PreprocessingPipeline:
    """
    A pipeline for preprocessing the raw data
    Args: 
        config (Dict[str, str]): Configuration dictionary containing preprocessing params
    """

    def __init__(self, config: Dict[str, str]):
        self.config = config
    
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to selected columns

        Args:
            pd.DataFrame: Input dataframe for dropping
        Returns:
            pd.DataFrame: Output dataframe after dropping selected columns
        """
        drop_cols = self.config.get("drop_columns", ["address"])
        df = df.drop(columns=drop_cols, errors="ignore")
        return df
    
    def handle_delimiter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to handling delimiter

        Args:
            pd.DataFrame: Input dataframe for handling delimeters
        Returns:
            pd.DataFrame: Output dataframe removing delimeters
        """
        features = self.config["features"]

        # Handle price column
        if features["target_column"] in df.columns:
            df[features["target_column"]] = df[features["target_column"]].apply(
                lambda x: float(str(x).replace("$", "").replace(",", "")) if pd.notnull(x) else None
            )

        # Handle land_size column
        if features["land_size"] in df.columns:
            df[features["land_size"]] = (
                df[features["land_size"]]
                .astype(str)                        # ensure string
                .str.replace("m²", "", regex=False) # remove "m²"
                .str.replace(",", "", regex=False)  # remove thousand separators
                .str.strip()                        # trim spaces
                .replace(["nan", "None", ""], np.nan)  # proper missing value
                .astype(float)                      # now safe
            )

        return df
    
    def change_float_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to reformatting some columns dtype

        Args:
            pd.DataFrame: Input dataframe for handling
        Returns:
            pd.DataFrame: Output dataframe after processed
        """
        features = self.config["features"]

        # Convert selected columns to float
        numeric_cols = [
            features["bedroom_nums"],
            features["bathroom_nums"],
            features["car_spaces"],
            features["lat"],
            features["lon"],
            features["postcode"],
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Encode city column (hardcoded mapping)
        if features["city"] in df.columns:
            df[features["city"]] = df[features["city"]].map({
                "Sydney": 0, "Newcastle": 1, "Wollongong": 2
            })

        return df
    
    def fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to filling not available value some columns dtype

        Args:
            pd.DataFrame: Input dataframe for handling
        Returns:
            pd.DataFrame: Output dataframe after processed
        """
        features = self.config["features"]

        for col in df.columns:
            if df[col].dtypes == "object": 
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                if col != features["city"]:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(0)
        return df
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline on the input DataFrame
        Args:
            pd.DataFrame: Input dataframe for handling
        Returns:
            pd.DataFrame: Output dataframe after processed
        """
        # Calling each of the processed step
        print(len(df))
        df = self.drop_columns(df)
        df = self.handle_delimiter(df)
        df = self.change_float_dtype(df)
        df = self.fill_na(df)
        return df
