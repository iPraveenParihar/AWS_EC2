import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import requests
import tensorflow as tf
import pickle

from flask import Flask, request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = None
tokenizer = None

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
		prediction = model.predict([data])

	return json.dumps(str(prediction))

def load_model():
	model = tf.keras.models.load_model('model/sms_spam_model.h5')

	with open('model/tokenizer.pickle','rb') as handle:
		tokenizer = pickle.load(handle)


if __name__ == "__main__":
	load_model()
	app.run(host='0.0.0.0', port=80, debug=True)