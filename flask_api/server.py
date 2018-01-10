import os
import pandas as pd
import dill as pickle
from flask import Flask, jsonify, request
#from utils import PreProcessing

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    """API Call
    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient='records')
    except Exception as e:
        raise e

    if test.empty:
        return bad_request()
    else:
        clf = 'model_v1.pk'
        loaded_model = load_model(clf)
        predictions = make_predictions(loaded_model, test)
        final_predictions = finalize_predictions(predictions)

        # return make_response(final_predictions)
        # what?!
        return "Test"


def load_model(model_name):
    print("Loading the model...")
    with open('./models/' + model_name, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


def make_predictions(model, x):
    print("The model has been loaded...doing predictions now...")
    return model.predict(x)


def finalize_predictions(predictions):
    return pd.DataFrame(list(pd.Series(predictions)))


def make_response(final_predictions):
    responses = jsonify(predictions=final_predictions.to_json(orient="records"))
    responses.status_code = 200
    return responses


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp
