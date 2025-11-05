from typing import Dict, Any
import pandas as pd
from common.utils import load_model
import shap
import matplotlib.pyplot as plt
import io, base64

class InferencePipeline:
    """
    A complete pipeline for making prediction using trained model.
    This class handles:
    - Loading a trained model
    - Preparing input data for inference
    - Making predictions
    - Post-processing predictions
    """

    def __init__(self, config: Dict[str, Any]) -> object:
        """
        Initializes the Inference pipeline with configuration data
        Args:
            config: Dict[str, Any]: Configuration dictionary
        """
        self.config = config
    

    def run(self, x: pd.DataFrame) -> dict:
        model = load_model(base_path=self.config['postprocessing']['model_path'])
        y_pred = model.predict(x)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)

        # top 3 features
        feature_importance = pd.DataFrame({
            "feature": x.columns,
            "shap_value": shap_values[0]  # first sample
        }).sort_values("shap_value", key=abs, ascending=False).head(3)

        return {
            "prediction": y_pred.tolist(),
            "top_features": feature_importance.to_dict(orient="records"),
            "shap_values": shap_values  # <-- add this
        }