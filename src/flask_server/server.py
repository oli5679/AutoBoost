import flask
import json
import pickle
import random
import pandas as pd
import datetime
import config
import logging
import numpy as np
import io
import sys

sys.path.append('src/model_builder')


"""
Rest API for querying model
"""


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "tools":
            renamed_module = "whyteboard.tools"

app = flask.Flask(__name__)

model = pickle.load(open(config.model_path, "rb"))


@app.errorhandler(404)
def not_found(error):
    return flask.make_response(flask.jsonify({"error": "Not found"}), 404)


@app.route("/")
def index():
    return "Welcome to my API"


@app.route("/score", methods=["POST"])
def generate_fraud_score():
    """
    JSON Args:
        request_payload (dict):
            model_features (dict) - model features to be scored
            

    Returns:
        JSON response (dict):
            model_inputs (dict): features passed to model
            model_score (numeric): evaluation of transaction's fraud risk,
            date_time (timestamp): time score returned
    """
    features_dict = json.loads(flask.request.data)
    assert list(features_dict.keys()) == config.model_feature_list
    features_array = np.array([list(features_dict.values())])
    global model
    fraud_pred = model.predict_proba(features_array)[0, 1]
    now = datetime.datetime.now()
    return json.dumps(
        {
            "model_inputs": features_dict,
            "model_score": fraud_pred,
            "date_time": now.timestamp(),
        }
    )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("loading model")

    logger.info("running app")
    app.run(debug=False, host="0.0.0.0")
