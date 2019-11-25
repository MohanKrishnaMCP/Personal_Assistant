import speech_recognition as sr 

r = sr.Recognizer()
m = sr.Microphone()

with m as source:
	print("Speak!! ")
	r.adjust_for_ambient_noise(source)
	#stop_listening = r.listen_in_background(m, callback)
	audio = r.listen(source)

	try:
		text = r.recognize_google(audio)
		print(f"{text}")
	except:
		print("Sorry could not recognize!!")
