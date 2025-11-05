import yaml
from pathlib import Path
from typing import Union, Any
import pickle

def read_config(path: Union[str, Path]) -> dict:
    """
    Reads a YAML configuration file and returns it as a dictionary

    Parameters:
        path: str or Path
            Path to YAML file.
    Returns: 
        dic: Parsed YAML content as a dictionary
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)


def save_model(model: Any, base_path: str) -> None:
    """
    Save model to the specified path
    Args:
        model: Trained model to save
        base_path: File path to save the trained model
    """
    path = Path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(base_path: str) -> Any:
    """
    Load model from the specific path
    Args:
        base_path: str: File path
    Returns:
        model: Any: Loaded model
    """
    path = Path(base_path)

    if not path.exists():
        raise FileNotFoundError(f"Model path not found {path}")

    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    return model