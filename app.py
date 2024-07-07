from flask import Flask, jsonify, request
from utils import status
from utils.evaluate_models import start_evaluation
from utils.make_recommendations import make_recommendations
from utils.train_models import start_training
from pprint import pprint
import numpy as np


app = Flask(__name__)


@app.route("/")
def home():
    with open("index.html") as file:
        return file.read()


@app.route("/api/data", methods=["POST"])
def get_data():
    ready = status.api_ready()
    if ready != "Model is ready":
        return jsonify({"message": ready}), 503
    received_data = request.json
    cinema_movies = received_data["movies"]
    user_ratings = received_data["user_ratings"]
    processed_data = make_recommendations(user_ratings, cinema_movies, False)
    # Umwandlung von int64 in int
    for item in processed_data:
        for key, value in item.items():
            if isinstance(value, np.int64):
                item[key] = int(value)

    return jsonify({"movies": processed_data})


@app.route("/api/train", methods=["POST"])
def train_model_endpoint():
    ready = status.api_ready()
    if ready != "Model is ready":
        return jsonify({"message": ready}), 503
    message = start_training()
    print(message)
    return jsonify({"message": message})


@app.route("/api/status", methods=["GET"])
def get_train_model_endpoint():
    message = status.api_ready()
    print(message)
    return jsonify({"message": message})


@app.route("/api/evaluate", methods=["POST"])
def evaluate_model_endpoint():
    ready = status.api_ready()
    if ready != "Model is ready":
        return jsonify({"message": ready}), 503
    message = start_evaluation()
    print(message)
    return jsonify({"message": message})


@app.route("/api/evaluate", methods=["GET"])
def get_evaluation_endpoint():
    ready = status.api_ready()
    if ready != "Model is ready":
        return jsonify({"message": ready}), 503
    (
        rmse_all,
        rmse_neighborhood,
        rmse_matrix_factorization,
        rmse_content_based,
        mae_all,
        mae_neighborhood,
        mae_matrix_factorization,
        mae_content_based,
    ) = status.get_evaluation_results()
    print(
        f"RMSE All: {rmse_all}, RMSE Neighborhood: {rmse_neighborhood}, RMSE Matrix Factorization: {rmse_matrix_factorization}, RMSE Content Based: {rmse_content_based}"
    )

    if rmse_all == 0:
        return (
            jsonify(
                {
                    "message": "No evaluation has been done yet. Pleaser trigger evaluation and then try again."
                }
            ),
            503,
        )

    return jsonify(
        {
            "Hybrid": {"RMSE": rmse_all, "MAE": mae_all},
            "Neighborhood": {"RMSE": rmse_neighborhood, "MAE": mae_neighborhood},
            "Matrix Factorization": {
                "RMSE": rmse_matrix_factorization,
                "MAE": mae_matrix_factorization,
            },
            "Content Based": {"RMSE": rmse_content_based, "MAE": mae_content_based},
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
