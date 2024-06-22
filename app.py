from flask import Flask, jsonify, request

app = Flask(__name__)

# So sollte der return am Ende aussehen
response_test = [
    {"movieId": 1, "score": 19},
    {"movieId": 2, "score": 85},
    {"movieId": 3, "score": 97},
    {"movieId": 4, "score": 5},
    {"movieId": 5, "score": 50},
    {"movieId": 6, "score": 90},
]
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


@app.route("/")
def home():
    return "<h1>Willkommen bei MoviemateAI!</h1><p>Hierbei handelt es sich um eine einfache REST-API, für ein Studentenprojekt der DHBW Mannheim. Alle weiteren Infos findest du auf unserem <a href='https://github.com/KI-Projekt/KI-Doku'>Github</a>!!</p>"


@app.route("/api/data", methods=["GET"])
def get_data():
    recieved_data = request.json
    print(recieved_data)
    return jsonify(response_test)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
