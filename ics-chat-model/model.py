from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.layers.core import Activation, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import argparse
import nltk
import itertools
from textGenModel import TextGenModel

data_path = "/Users/dhruvjhamb/Projects/intelli-bot/ics-chat-model/data"

# constant token and params for our models
START_TOKEN = "SENTENCE_START"
END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
PADDING_TOKEN = "PADDING"

vocab_size = 5000
sent_max_len = 20

def load_data(filename):
	with tf.gfile.GFile(filename, "r") as f:
		return f.read().replace("\n", "<eos>").split()

def word_tokenization(data):
	sentences = [[START_TOKEN] + nltk.word_tokenize(entry.lower()) + [END_TOKEN] for entry in data]
	return sentences

# creates index_to_word and word_to_index mappings, given the data and a max vocabulary size
def get_words_mappings(tokenized_sentences, vocabulary_size):
    # we can rely on nltk to quickly get the most common words, and then limit our vocabulary to the specified size
    frequence = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = frequence.most_common(vocabulary_size)
    index_to_word = [x[0] for x in vocab]
    # Add padding for index 0
    index_to_word.insert(0, PADDING_TOKEN)
    # Append unknown token (with index = vocabulary size + 1)
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    return index_to_word, word_to_index

def main():
	# load training data set
	filename = os.path.join(data_path, "friends.txt")
	train_data = load_data(filename)

	# word tokenization for each sentence, while adding start and end tokens
	sentences = word_tokenization(train_data[:20000])

	# get mappings and update vocabulary size
	index_to_word, word_to_index = get_words_mappings(sentences, vocab_size)
	vocabulary_size = len(index_to_word)

	# Generate training data by converting tokenized sentenced to indexes (and replacing unknown words)
	train_size = min(len(sentences), 100000)
	train_data = [[word_to_index.get(w,word_to_index[UNKNOWN_TOKEN])  for w in sent] for sent in sentences[:train_size]]

	# pad sentences to fixed lenght (pad with 0s if shorter, truncate if longer)
	train_data = sequence.pad_sequences(train_data, maxlen=sent_max_len, dtype='int32', padding='post', truncating='post')

	# create training data for rnn: 
	# input is sentence truncated from last word, output is sentence truncated from first word
	X_train = train_data[:,:-1]
	y_train = train_data[:,1:]
	#X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], 1])
	y_train = y_train.reshape([y_train.shape[0], y_train.shape[1], 1]) # needed cause out timedistributed layer

	# Define model and parameters
	hidden_size = 512
	embedding_size = 128

	# model with embedding
	'''model = Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, mask_zero=True))
	# add batch norm
	#model.add(TimeDistributed(Flatten()))
	model.add(LSTM(hidden_size, return_sequences=True, activation='relu'))
	model.add(TimeDistributed(Dense(vocabulary_size, activation='softmax')))
	model.summary()

	# recompile also if you just want to keep training a model just loaded from memory
	loss = 'sparse_categorical_crossentropy'
	optimizer = 'adam'
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

	# Train model
	num_epoch = 5
	batch_size = 32
	model.fit(X_train, y_train, epochs=num_epoch, batch_size=batch_size, verbose=1)

	# export model
	model.save(data_path + "/model.hdf5")'''
	model = load_model(data_path + "/model.hdf5") # -" + str(num_epoch) + ".hdf5")

	parser = argparse.ArgumentParser()
	parser.add_argument('input_string', type=str, help='The input message')
	args = parser.parse_args()
	input_list = args.input_string.split(' ')

	sentences = [word_tokenization(i) for i in input_list]

	bot_response = []

	for i in range(len(input_list)):
		input_data = [[word_to_index.get(w,word_to_index[UNKNOWN_TOKEN])  for w in sent] for sent in sentences[i]]

		input_data = sequence.pad_sequences(input_data, maxlen=sent_max_len, dtype='int32', padding='post', truncating='post')

		prediction = model.predict(input_data[0])
		prediction = np.argmax(prediction)
		word = ''
		try:
			word = index_to_word[prediction]
		except:
			word = 'unknown_word'
		bot_response.append(word)
	
	bot_response_string = ''
	for word in bot_response:
		if word != 'unknown_word':
			bot_response_string += word + ' '
	if not bot_response_string:
		bot_response_string = 'I am confused.'
	print(bot_response_string)

	'''text_gen = TextGenModel(model, index_to_word, word_to_index, sent_max_len=sent_max_len, 
                                    temperature=1.0,
                                    use_embeddings=True)
	# generate 1 new sentence
	n_sents = 1
	for _ in range(n_sents):
		res = text_gen._generate_sentence(1, input_data[0])
		print(res)'''
	    #res = text_gen.pretty_print_sentence(text_gen.get_sentence(15))

if __name__== "__main__":
    main()
