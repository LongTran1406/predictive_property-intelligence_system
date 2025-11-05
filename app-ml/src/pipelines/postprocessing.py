from typing import Dict, Any
import pickle
from common.utils import save_model
import pandas as pd

class PostProcessingPipeline:
    """
    A pipeline handling postprocessing step in the machine learning pipeline:
    - Saves trained model
    - Formatting and returning prediction results
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializing the postprocessing pipeline 
        Args:
            config: Dict[str, Any]: Dictionary containing all configuration information
        """

        self.config = config
    
    def run_train(self, model: Any) -> None:
        """
        Save the trained model to the file path specified in the config
        Args:
            model: Any: Trained machine learning model.
        Returns: None
        """

        model_path = self.config['postprocessing']['model_path']
        save_model(model, model_path)
    
    def run_inference(self, y_pred: float) -> pd.DataFrame:
        """
        Formatting the model prediction
        Args:
            y_pred: float: predict value
        Returns:
            pd.DataFrame: formatted output
        """
        return y_pred
        



