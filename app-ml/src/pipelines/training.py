import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


class TrainingPipeline:
    """
    A complete pipeline for training and optimizing machine learning model
    using GridSearch
    Args:
        config (Dict[str, Any]): Configuration dictionary with training pipelien
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare dataset by dropping unused features and spliting data into train/valid/test sets
        Args:
            pd.DataFrame: Input DataFrame
        Returns:
            pd.DataFrame: train features set
            pd.DataFrame: validation features set
            pd.DataFrame: test features set
            pd.DataFrame: train target set
            pd.DataFrame: validation target set
            pd.DataFrame: test target set
        """
        target_col = self.config['features']['target_column']
        X, y = df.drop(columns = [target_col, 'price_per_m2']), df[target_col]

        train_size = self.config['training']['train_size']
        validation_size = self.config['training']['validation_size']
        test_size = self.config['training']['test_size']

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = validation_size / (validation_size + train_size), random_state = 42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame) -> Any:
        """
        Function performing training (using mlflow)

        Args:
            pd.DataFrame: X_train, X_val, X_test, y_train, y_val, y_test
        Returns:
            Any: trained model

        """

        mlflow.set_tracking_uri(self.config.get("mlflow_uri", "http://127.0.0.1:5000/"))

        with mlflow.start_run() as parent_run:
            # --- Linear Regression ---
            with mlflow.start_run(run_name="LinearRegression", nested=True):
                lr = LinearRegression()
                lr.fit(X_train, y_train)

                mlflow.log_metric("train_r2", lr.score(X_train, y_train))
                mlflow.log_metric("val_r2", lr.score(X_val, y_val))
                mlflow.log_metric("test_r2", lr.score(X_test, y_test))

                preds_lr = lr.predict(X_test)
                rmse_lr = np.sqrt(mean_squared_error(y_test, preds_lr))
                mlflow.log_metric("test_rmse", rmse_lr)

                mlflow.sklearn.log_model(lr, "linear_model")

            # --- XGBoost + GridSearch ---
            param_grid = {
                'n_estimators': [100, 200, 300],
            }

            xgb_model = xgb.XGBRegressor(
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=15,
                reg_lambda=35,
                random_state=42
            )

            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=5,
                verbose=1,
                n_jobs=-1
            )

            with mlflow.start_run(run_name="XGBoost", nested=True):
                grid_search.fit(X_train, y_train)
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("best_cv_neg_mse", grid_search.best_score_)

                best_model = grid_search.best_estimator_
                preds_xgb = best_model.predict(X_test)
                rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
                mlflow.log_metric("test_rmse", rmse_xgb)

                mlflow.sklearn.log_model(best_model, "xgboost_model")

        return best_model

        
    
    def run(self, df: pd.DataFrame) -> Any:
        """
        Run the full training pipeline
        Args:
            pd.DataFrame: Input data frame
        Returns:
            Any: Trained Model
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_dataset(df)
        models = self.train(X_train, X_val, X_test, y_train, y_val, y_test)
        return models