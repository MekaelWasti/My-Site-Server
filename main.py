from flask import Flask
from flask import request
from flask import jsonify
from flask_sslify import SSLify
from flask_cors import CORS
from HDR import HDR
import ssl

# Certificate

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
# context.load_cert_chain('./certification/SELF_SIGNED/certificate.crt', './certification/SELF_SIGNED/private.key')
context.load_cert_chain('./certification/ZERO_SSL/certificate.crt', './certification/ZERO_SSL/private.key')



app = Flask(__name__)
sslify = SSLify(app)
# app.config['VERIFY_REQUESTS'] = False
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
    # app.run(host="0.0.0.0", port=5001)
    app.run(host="0.0.0.0", port=63030, ssl_context=context, threaded=True)
    # app.run(host="0.0.0.0", port=10201, ssl_context=context, threaded=True)
    # app.run(host="0.0.0.0", port=80, ssl_context=context, threaded=True)
