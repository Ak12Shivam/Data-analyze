from flask import Flask, request, render_template, send_file, jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# Ensure the "static" folder exists for saving images
if not os.path.exists("static"):
    os.makedirs("static")

data = None  # Global variable to store the cleaned data

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    global data
    if request.method == "POST":
        try:
            # Check if file is present
            if 'file' not in request.files:
                logging.error("No file part in the request")
                return jsonify({"error": "No file part in the request"}), 400

            file = request.files["file"]
            if not file or file.filename == '':
                logging.error("No file selected")
                return jsonify({"error": "No file selected"}), 400

            # Validate file type and size
            if not allowed_file(file.filename):
                logging.error(f"Unsupported file format: {file.filename}")
                return jsonify({"error": "Unsupported file format! Please upload CSV or Excel."}), 400

            filename = secure_filename(file.filename)
            logging.info(f"Received file: {filename}")

            # Load the uploaded file
            try:
                if filename.endswith(".csv"):
                    data = pd.read_csv(file)
                else:
                    data = pd.read_excel(file)
            except Exception as e:
                logging.error(f"Error reading file {filename}: {str(e)}")
                return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

            # Check if data is empty
            if data.empty:
                logging.error("Uploaded file is empty")
                return jsonify({"error": "Uploaded file is empty"}), 400

            # Data Cleaning
            clean_option = request.form.get("clean_option")
            if clean_option == "drop_na":
                data = data.dropna()
            elif clean_option == "fill_mean":
                data = data.fillna(data.mean(numeric_only=True))
            elif clean_option == "fill_median":
                data = data.fillna(data.median(numeric_only=True))
            elif clean_option == "fill_mode":
                data = data.fillna(data.mode().iloc[0])
            else:
                logging.error(f"Invalid cleaning option: {clean_option}")
                return jsonify({"error": "Invalid cleaning option selected"}), 400

            # Data Summary
            summary = data.describe().to_html(classes="table-auto w-full border-collapse border border-gray-300")
            null_info = data.isnull().sum().to_frame("Missing Values")
            null_info["Data Type"] = data.dtypes
            null_info_html = null_info.to_html(classes="table-auto w-full border-collapse border border-gray-300")

            # Visualization
            image_paths = []
            graph_size = request.form.get("graph_size", "medium")
            graph_option = request.form.getlist("graph_option")
            graph_size_dict = {"small": (8, 4), "medium": (10, 6), "large": (12, 8)}

            if "histogram" in graph_option:
                try:
                    plt.figure(figsize=graph_size_dict[graph_size])
                    data.hist(bins=20, figsize=graph_size_dict[graph_size], edgecolor="black")
                    plt.tight_layout()
                    histogram_path = "static/histogram.png"
                    plt.savefig(histogram_path)
                    plt.close()
                    image_paths.append(histogram_path)
                except Exception as e:
                    logging.error(f"Error generating histogram: {str(e)}")
                    image_paths.append(None)

            if "heatmap" in graph_option:
                try:
                    plt.figure(figsize=graph_size_dict[graph_size])
                    numeric_data = data.select_dtypes(include=['float64', 'int64'])
                    if not numeric_data.empty:
                        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
                        heatmap_path = "static/heatmap.png"
                        plt.savefig(heatmap_path)
                        plt.close()
                        image_paths.append(heatmap_path)
                    else:
                        logging.warning("No numeric data for heatmap")
                        image_paths.append(None)
                except Exception as e:
                    logging.error(f"Error generating heatmap: {str(e)}")
                    image_paths.append(None)

            if "boxplot" in graph_option:
                try:
                    plt.figure(figsize=graph_size_dict[graph_size])
                    sns.boxplot(data=data.select_dtypes(include=['float64', 'int64']))
                    boxplot_path = "static/boxplot.png"
                    plt.savefig(boxplot_path)
                    plt.close()
                    image_paths.append(boxplot_path)
                except Exception as e:
                    logging.error(f"Error generating boxplot: {str(e)}")
                    image_paths.append(None)

            if "pairplot" in graph_option:
                try:
                    pairplot = sns.pairplot(data.select_dtypes(include=['float64', 'int64']))
                    pairplot_path = "static/pairplot.png"
                    pairplot.savefig(pairplot_path)
                    plt.close()
                    image_paths.append(pairplot_path)
                except Exception as e:
                    logging.error(f"Error generating pairplot: {str(e)}")
                    image_paths.append(None)

            # Filter out None values from image_paths
            image_paths = [path for path in image_paths if path is not None]

            # Advanced Analysis
            advanced_analysis = ""
            if "feature_importance" in graph_option:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100)
                    data_numeric = data.select_dtypes(include=["float64", "int64"]).dropna()
                    if not data_numeric.empty and data_numeric.shape[1] > 1:
                        X = data_numeric.iloc[:, :-1]
                        y = data_numeric.iloc[:, -1]
                        model.fit(X, y)
                        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
                        advanced_analysis += f"<h2>Feature Importance:</h2>{feature_importances.sort_values(ascending=False).to_frame('Importance').to_html(classes='table-auto w-full border-collapse border border-gray-300')}"
                    else:
                        advanced_analysis += "<p>Not enough numeric data for Feature Importance analysis.</p>"
                except Exception as e:
                    logging.error(f"Feature Importance Error: {str(e)}")
                    advanced_analysis += f"<p>Feature Importance analysis failed: {str(e)}</p>"

            if "regression" in graph_option:
                try:
                    from sklearn.linear_model import LinearRegression
                    data_numeric = data.select_dtypes(include=["float64", "int64"]).dropna()
                    if not data_numeric.empty and data_numeric.shape[1] > 1:
                        model = LinearRegression()
                        X = data_numeric.iloc[:, :-1]
                        y = data_numeric.iloc[:, -1]
                        model.fit(X, y)
                        regression_score = model.score(X, y)
                        advanced_analysis += f"<h2>Linear Regression R² Score: {regression_score:.2f}</h2>"
                    else:
                        advanced_analysis += "<p>Not enough numeric data for Regression analysis.</p>"
                except Exception as e:
                    logging.error(f"Regression Error: {str(e)}")
                    advanced_analysis += f"<p>Regression analysis failed: {str(e)}</p>"

            # Prepare response for AJAX
            response = {
                "summary": summary,
                "null_info": null_info_html,
                "image_paths": image_paths,
                "advanced_analysis": advanced_analysis if advanced_analysis else "No advanced analysis available."
            }
            logging.info("Request processed successfully")
            return jsonify(response), 200

        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    return render_template("index.html")

@app.route("/download_cleaned")
def download_cleaned():
    global data
    if data is None:
        logging.error("No data available for download")
        return jsonify({"error": "No data available to download!"}), 400

    try:
        output = io.BytesIO()
        data.to_csv(output, index=False)
        output.seek(0)
        logging.info("Cleaned data downloaded successfully")
        return send_file(output, as_attachment=True, download_name="cleaned_data.csv", mimetype="text/csv")
    except Exception as e:
        logging.error(f"Error generating download: {str(e)}")
        return jsonify({"error": "Failed to generate download file."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)