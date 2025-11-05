from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for server environments
import matplotlib.pyplot as plt
import shap
import os
import sys
from pathlib import Path
import uuid
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root)
print(sys.path)

from pipelines.pipeline_runner import PipelineRunner
from common.utils import read_config
from common.data_manager import DataManager

app = Flask(__name__)

# Load the model once
config_path = project_root / 'config' / 'config.yaml'
config = read_config(config_path)
inference  = PipelineRunner(config = config, data_manager = DataManager(config = config))

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/prediction")
def prediction():
    return render_template('prediction.html')

@app.route("/dataset_insights")
def dataset_insights():
    """
    Serve the Dataset Insights page template.
    """
    return render_template("analysis.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received JSON:", data)  # Debug input
        
        if not data:
            return jsonify({"error": "No input data received"}), 400

        df = pd.DataFrame([data])
        print("DataFrame created:\n", df)
        df.to_parquet('data/test/Test.parquet', index=False)
        print("Saved test parquet at data/test/Test.parquet")

        # Run inference
        results, df = inference.run_inference()
        print("Inference results:", results)
        if "shap_values" not in results:
            return jsonify({"error": "SHAP values missing in inference results"}), 500

        shap_values = results.get("shap_values")
        print("SHAP values type:", type(shap_values), "shape/length:", np.shape(shap_values))

        # Mapping for nicer feature names
        feature_name_map = {
            "bedroom_nums": "Bedrooms",
            "bathroom_nums": "Bathrooms",
            "car_spaces": "Car Spaces",
            "land_size": "Land Size (m²)",
            "lat": "Latitude",
            "lon": "Longitude",
            "postcode": "Postcode",
            "city": "City",
            "dist_to_city": "Distance to City (km)",
            "avg_price_by_postcode": "Avg Price by Postcode",
            "postcode_avg_price_per_m2": "Avg Price/m² by Postcode"
        }

        # Check df columns vs SHAP columns
        print("DataFrame columns:", df.columns.tolist())
        if len(shap_values[0]) != len(df.columns):
            print("Warning: SHAP values length does not match input features!")

        # Create SHAP Explanation
        expl = shap.Explanation(
            values=shap_values[0],
            base_values=None,
            data=df.iloc[0],
            feature_names=df.columns
        )

        # Sort features by absolute impact
        indices = np.argsort(np.abs(expl.values))[::-1]
        sorted_features = [feature_name_map.get(expl.feature_names[i], expl.feature_names[i]) for i in indices]
        sorted_values = expl.values[indices]
        colors = ['#10b981' if v > 0 else '#ef4444' for v in sorted_values]

        print("Top features sorted:", sorted_features)
        print("Top SHAP values:", sorted_values)

        # Take top N features
        top_n = min(10, len(sorted_features))
        sorted_features = sorted_features[:top_n]
        sorted_values = sorted_values[:top_n]
        colors = colors[:top_n]

        abs_sum = np.sum(np.abs(sorted_values))
        if abs_sum == 0:
            print("Warning: Sum of absolute SHAP values is zero, cannot convert to percentages")
            percent_values = [0] * top_n
        else:
            percent_values = [v / abs_sum * 100 for v in sorted_values]

        print("Percentage contributions:", percent_values)

        # Plot
        shap_plot_file = f"shap_plot_{uuid.uuid4().hex}.png"
        shap_folder = project_root / "static" / "shap_plot"
        shap_folder.mkdir(exist_ok=True, parents=True)

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0f1724')
        ax.set_facecolor('#0f1724')
        bars = ax.barh(sorted_features, percent_values, color=colors, edgecolor='white', height=0.6)
        ax.set_xlabel("Contribution to total impact (%)", color='white')
        ax.set_title("Top Feature Contributions", color='white')
        ax.invert_yaxis()
        pad = 1
        max_bar = max(percent_values) if percent_values else 1
        for bar, value in zip(bars, percent_values):
            y = bar.get_y() + bar.get_height() / 2
            x = max_bar + pad
            ax.text(x, y, f"{value:.1f}%", va='center', ha='left', fontsize=10, color='white')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.tick_params(colors='white', labelsize=10)
        plt.tight_layout()
        plt.savefig(shap_folder / shap_plot_file, dpi=150, facecolor='#0f1724')
        plt.close()
        print("SHAP plot saved at:", shap_folder / shap_plot_file)

        response = {
            "prediction": results.get("prediction"),
            "top_features": results.get("top_features"),
            "shap_plot": f"/static/shap_plot/{shap_plot_file}",
            "data": df.iloc[0].to_dict()
        }
        print("Returning response:", response)
        return jsonify(response)

    except Exception as e:
        import traceback
        print("Error in /predict:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    
@app.route("/analysis_plot", methods=["POST"])
def analysis_plot():
    """
    Return summary stats, distribution plot for a selected feature,
    and average value by postcode for numeric features.
    """
    try:
        data = request.json
        print(data)
        feature = data.get("feature")
        if feature is None:
            return jsonify({"error": "Feature not specified"}), 400

        # Load dataset
        path = project_root / 'data' / 'ml-ready' / 'database_ml.parquet'
        df = DataManager.load_data(path)  # assumes this method exists
        if df is None or df.empty:
            return jsonify({"error": "Dataset could not be loaded"}), 500

        if feature not in df.columns:
            return jsonify({"error": f"Feature '{feature}' not found"}), 400

        # Summary stats
        if pd.api.types.is_numeric_dtype(df[feature]):
            summary = df[feature].describe().to_dict()
            summary = {k: round(v, 2) for k,v in summary.items()}
        else:
            summary = df[feature].value_counts().head(10).to_dict()

        # Plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        import uuid

        plot_file = f"analysis_{feature}_{uuid.uuid4().hex}.png"
        plot_folder = project_root / "static" / "plots"
        plot_folder.mkdir(exist_ok=True, parents=True)

        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0f1724')
        ax.set_facecolor('#0f1724')
        if pd.api.types.is_numeric_dtype(df[feature]):
            sns.histplot(df[feature], kde=True, ax=ax, color='#10b981')
        else:
            df[feature].value_counts().head(10).plot(kind='bar', ax=ax, color='#10b981')
            ax.set_ylabel("Count")

        ax.set_title(f"{feature} Distribution", color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        plt.tight_layout()
        plt.savefig(plot_folder / plot_file, dpi=150, facecolor='#0f1724')
        plt.close()

        # Average by postcode for numeric features
        avg_by_postcode = {}
        if "postcode" in df.columns and pd.api.types.is_numeric_dtype(df[feature]) and feature != "postcode":
            grouped = df.groupby("postcode")[feature].mean().round(2)
            avg_by_postcode = grouped.to_dict()  # all postcodes; frontend filters by user input

        return jsonify({
            "summary": summary,
            "plot_url": f"/static/plots/{plot_file}",
            "avg_by_postcode": avg_by_postcode
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/postcode_avg", methods=["POST"])
def postcode_avg():
    """
    Return average values of all numeric features for a given postcode.
    """
    try:
        data = request.json
        postcode = str(data.get("postcode")).strip()
        if not postcode:
            return jsonify({"error": "Postcode not specified"}), 400

        # Load dataset
        path = project_root / 'data' / 'ml-ready' / 'database_ml.parquet'
        df = DataManager.load_data(path)

        if "postcode" not in df.columns:
            return jsonify({"error": "Postcode column not found"}), 400

        # Filter dataset by postcode
        postcode = float(postcode)
        filtered = df[df["postcode"].astype(float) == postcode]
        if filtered.empty:
            return jsonify({"error": "No data for this postcode"}), 404

        # Map feature names
        feature_name_map = {
            "bedroom_nums": "Bedrooms",
            "bathroom_nums": "Bathrooms",
            "car_spaces": "Car Spaces",
            "land_size": "Land Size (m²)",
            "lat": "Latitude",
            "lon": "Longitude",
            "postcode": "Postcode",
            "city": "City",
            "dist_to_city": "Distance to City (km)",
            "avg_price_by_postcode": "Avg Price by Postcode",
            "postcode_avg_price_per_m2": "Avg Price/m² by Postcode"
        }

        # Compute averages for numeric features
        numeric_cols = filtered.select_dtypes(include='number').columns.tolist()
        avg_dict = filtered[numeric_cols].mean().round(2).to_dict()

        # Map column keys to human-readable names
        avg_dict_mapped = {feature_name_map.get(k, k): v for k, v in avg_dict.items()}

        return jsonify({"avg_by_postcode": avg_dict_mapped})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/scatter_plot", methods=["POST"])
def scatter_plot():
    """
    Return a scatterplot of price vs selected feature.
    """
    try:
        data = request.json
        feature = data.get("feature")
        if feature is None:
            return jsonify({"error": "Feature not specified"}), 400

        path = project_root / 'data' / 'ml-ready' / 'database_ml.parquet'
        df = DataManager.load_data(path)
        
        if feature not in df.columns:
            return jsonify({"error": f"Feature '{feature}' not found"}), 400
        if "price" not in df.columns:
            return jsonify({"error": "Dataset missing 'price' column"}), 400

        plot_file = f"scatter_{feature}_{uuid.uuid4().hex}.png"
        plot_folder = project_root / "static" / "plots"
        plot_folder.mkdir(exist_ok=True, parents=True)

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0f1724')
        ax.set_facecolor('#0f1724')
        sns.scatterplot(x=df[feature], y=df["price"], ax=ax, color='#10b981')
        ax.set_xlabel(feature, color='white')
        ax.set_ylabel("Price", color='white')
        ax.set_title(f"Price vs {feature}", color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        plt.tight_layout()
        plt.savefig(plot_folder / plot_file, dpi=150, facecolor='#0f1724')
        plt.close()

        return jsonify({"plot_url": f"/static/plots/{plot_file}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
