from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
from HDR import HDR

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def hello():
    return {'message': 'Hello'}


@app.route('/predict', methods=['POST'])
def predict():
    # Handle the incoming request and perform necessary actions
    # (e.g., process the image, run it through the deep learning model, generate a prediction)

    data = request.json
    predictedDigit = HDR(data)

    # Return a response to the React app
    return jsonify({'result': predictedDigit})


if __name__ == '__main__':
    app.run(port=5000)
