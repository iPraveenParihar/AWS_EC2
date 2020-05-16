import json
import requests

from flask import Flask, request

import model.predict

app = Flask(__name__)

@app.route('/')
def index():
	return "Index Page"

@app.route('/predict', methods=['GET','POST'])
def get_predict():
	data = request.form.get('data')
	if data == None:
		return 'Got None'
	else:
		prediction = model.predict.predict([data])

	return json.dumps(str(prediction))


if __name__ == "__main__":
	app.run(host='127.0.0.1', debug=True)