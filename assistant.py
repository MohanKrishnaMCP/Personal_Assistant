import playsound
#import os
import time
import speech_recognition as sr
from gtts import gTTS


def speak(text):
	tts = gTTS(text = text, lang = "en")
	filename = "voice.mp3"
	tts.save(filename)
	playsound.playsound(filename)


def get_audio():
	r = sr.Recognizer()
	m = sr.Microphone()

	with m as source:
		print("Speak!! ")
		r.adjust_for_ambient_noise(source)
		audio = r.listen(source)
		said = ""

		try:
			said = r.recognize_google(audio)
			print(f"{said}")
		except:
			print("Sorry could not recognize!!")

	return said


text = get_audio()

if "hello" in text:
	speak("hello, Nigga!")