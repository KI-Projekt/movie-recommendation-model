# Movie-Recommendation-Model

In the context of a student project we want to develop a movie recommendation system.

## Different Approaches


We train 3 models with different approaches:
- Content-Based
- Collaborative-Filtering Neighborhood
- Collaborative-Filtering Matrix Factorization

Each model returns a score for the recommended movies. These scores from the different models can be combined for a overall recommendation.

## Start Server
Install flask with the following command:

```bash
pip install flask  
```

Start the Server with the following command:
```bash
export FLASK_APP=app
flask run
```
If this does not work try the following command:
```bash
set FLASK_APP=app
python -m flask run
```

Open the website for the documentation of the enpoints.