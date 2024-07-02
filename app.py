from flask import Flask, jsonify, request
from utils.make_recommendations import process_data
from utils.train_models import train_model


app = Flask(__name__)

# Beispiel-Response
response_test = [
    {"externalId": 1, "movieTitle": "Test", "year": 2018, "score": 19},
    {"externalId": 2, "movieTitle": "Test2", "year": 2019, "score": 85},
    {"externalId": 3, "movieTitle": "Test3", "year": 2020, "score": 97},
    {"externalId": 4, "movieTitle": "Test4", "year": 2021, "score": 5},
    {"externalId": 5, "movieTitle": "Test5", "year": 2022, "score": 50},
    {"externalId": 6, "movieTitle": "Test6", "year": 2023, "score": 90},
]


@app.route("/")
def home():
    with open("index.html") as file:
        return file.read()


@app.route("/api/data", methods=["POST"])
def get_data():
    received_data = request.json
    print(received_data)
    processed_data = process_data(received_data)
    return jsonify(processed_data)


@app.route("/api/train", methods=["POST"])
def train_model_endpoint():
    received_data = request.json
    print(received_data)
    message = train_model(received_data)
    return jsonify({"message": message})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# So kommen die Daten am Endpunkt an
#
# Test_requerst = {
#     "movies": [
#         {
#             "movieId": 1,
#             "title": "Toy Story (1995)",
#             "genres": "Adventure, Animation, Children, Comedy, Fantasy",
#             "description": "Movie description",
#         },
#         {
#             "movieId": 2,
#             "title": "Jumanji (1995)",
#             "genres": "Adventure, Children, Fantasy",
#             "description": "Movie description",
#         }
#     ],
#     "user":{
#         "genres": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
#         "ratings": [
#             {"movieTitle": 1, "rating": 5},
#             {"movieTitle": 2, "rating": 4.5}
#         ]
#     }
# }
