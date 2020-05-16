import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import requests
import tensorflow as tf
import pickle

from flask import Flask, request, render_template, url_for
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = None
tokenizer = None
prediction_result = None

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

@app.route('/', methods=['GET','POST'])
def index():
	
	if request.method == "GET":
		return render_template("index.html")	

	if request.method == "POST":
		model_name = request.form['model_name']
		action = request.form['action']

		return action

@app.route('/about')
def about():
	return render_template("about.html")

@app.route('/sms_spam', methods=['GET', 'POST'])
def sms_spam():
	
	if request.method == "GET":
		return render_template("sms_spam.html", prediction = None)

	elif request.method == "POST":
		data = request.form['sms_input_data']
		prediction_result = get_prediction([data])
		return render_template("sms_spam.html", prediction = prediction_result, data = data)

@app.route('/arch')
def model_summary():
	return model.summary()


def get_prediction(data):

	global model
	global tokenizer

	try:
		model = tf.keras.models.load_model('./model/sms_spam_model.h5')

		with open('./model/tokenizer.pickle','rb') as handle:
			tokenizer = pickle.load(handle)

	except Exception:
		print(Exception)

	seq = tokenizer.texts_to_sequences(data)
	pad = pad_sequences(seq, maxlen=38)

	prediction = model.predict(pad)

	if(prediction.round() == 1):
		return True
	else:
		return False 

if __name__ == "__main__":
	app.run(host='127.0.0.1', debug=True)