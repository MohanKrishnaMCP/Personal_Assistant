import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import playsound
import time
import os
#from tempfile import TemporaryFile
from mutagen.mp3 import MP3
from time import sleep
import speech_recognition as sr
from gtts import gTTS
import pyglet

import numpy
import tflearn
import tensorflow
#import pickle
import random
import json

with open("intent.json") as file:
	data = json.load(file)

# try:
# 	with open("dat.pickle","rb") as f:
# 		words,labels,training,output =pickle.load(f)

#except:
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"])

# print(docs_x)
# print(docs_y)
# print(labels)
# print(words)

words = [stemmer.stem(w.lower()) for w in words if w!="?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
	bag = []
	wrds = [stemmer.stem(w) for w in doc]
	print(x,wrds)

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)

	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)
print (output)

	#with open("dat.pickle","wb") as f:
	#	pickle.dump((words,labels,training,output),f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#try:
#	model.load("model.tflearn")
#except:
model.fit(training,output, n_epoch=1000, batch_size=8,show_metric=True)
model.save("model.tflearn")


def bagwrds(s,words):
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for ss in s_words:
		for i,w in enumerate(words):
			if w == ss:
				bag[i] = 1
	return numpy.array(bag)


def speak(text):
	tts = gTTS(text = text, lang = "en")
	filename = "voice.mp3"
	tts.save(filename)

	# music = pyglet.media.load(filename, streaming = False)
	# music.play()
	playsound.playsound(filename)
	audio = MP3('voice.mp3')
	sleep(audio.info.length + 10)
	os.remove(filename)
	# f = TemporaryFile()
	# tts.write_to_fp(f)
	# f.close()


def get_audio():
	r = sr.Recognizer()
	m = sr.Microphone()

	with m as source:
		print("You: ")
		r.adjust_for_ambient_noise(source)
		audio = r.listen(source)
		said = ""

		try:
			said = r.recognize_google(audio)
			print(f"{said}")
		except:
			print("Sorry could not recognize!!")

	return said


def chat():
	print("ask something (q to exit)")
	while True:

		inp = get_audio()
		# inp = input("You: ")
		# if inp.lower == "q":
		# 	break
		results = model.predict([bagwrds(inp,words)])[0]
		results_index = numpy.argmax(results)
		tag = labels[results_index]
		
		if results[results_index]>0.7:

			for tg in data["intents"]:
				if tg["tag"] == tag:
					responses = tg["responses"]

			reply = random.choice(responses)
			speak(reply)
			print(reply)

		else:
			speak("Come again Nigga!")
			print("Come again Nigga!")


chat()
