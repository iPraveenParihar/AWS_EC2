import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def predict(data):
	model = tf.keras.models.load_model('model/sms_spam_model.h5')

	with open('model/tokenizer.pickle','rb') as handle:
		tokenizer = pickle.load(handle)

	seq = tokenizer.texts_to_sequences(data)
	pad = pad_sequences(seq, maxlen=38)

	prediction = model.predict(pad)

	if(prediction.round() == 1):
		return True
	else:
		return False