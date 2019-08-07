#!/usr/bin/env python
# coding: utf-8

# In[39]:


import os
import speech_recognition as sr
from subprocess import Popen, PIPE
from gtts import gTTS
import time


def listen():
    # obtain audio from the microphone
    r = sr.Recognizer()
    r.dynamic_energy_threshold = False
    
    for i in range(0,50) :
        with sr.Microphone() as source:
            print("Say yes or no \n")
            #time.sleep(2)
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source,  timeout= 10.0)  
                # recognize speech using google
                try:
                    #print("Say yes or no \n")
                    text = r.recognize_google(audio)
                    print(text)

                    if "yes" in text or "start" in text:
                        return True
                    elif "no" in text:
                        return False
                    else:
                        print("please respond with just a yes or a no\n")
                        speak_g( "please respond with just a yes or a no")

                    #print("google thinks you said " + r.recognize_google(audio))
                except sr.UnknownValueError:
                    print("I could not understand audio\n")
                    speak_g("come again, I could not understand what you said\n")

                except sr.RequestError as e:
                    print("sorry i have a poor internet connection\n")       
                                
            except sr.WaitTimeoutError:
                print(" sorry I timed out")
                speak_g(" sorry I timed out, please be quick, and say yes or no")
           
            
    return False

                  
def speak_g(text):
    
    speech = gTTS(text)
    speech.save('audio.mp3')
    os.system('mpg123 audio.mp3')
    return
    
def speak(dialogue_number, part_number):
    
    os. chdir("/home/sonu/audio_files_SARA"+"/dialogue_" + str(dialogue_number))
    path = "mpg123" + " " + "dialogue_" + str(dialogue_number) + "_part" + str(part_number) + ".mp3"
    os.system(path)
    os. chdir("/home/sonu")
    return

def dialogue_0(value):
    if value:
        speak(0,2)
    elif value:
        speak(0,3)
    return


def dialogue_1(value):
    if value:
        speak(1,1)
    else :
        speak(1,2)
    return


def dialogue_2(value):
    if value:
        #starting localization node
        global proc
        proc = Popen(['xterm', '-e', 'roslaunch kangaroo_driver wheelchair_rtabmap_localization.launch'])
        speak(2,1)
        time.sleep(15)
        speak(2,2)
    else :
        speak(2,3)

    return

def dialogue_3(value):
    if value:
        speak(3,1)
    else :
        speak(3,2)
        proc.kill()
        proc = Popen(['xterm', '-e', 'roslaunch kangaroo_driver wheelchair_rtabmap_localization.launch'])        
    return

def dialogue_4(value):
    if value:
        speak(4,1)
    else :
        speak(4,2)
    return
        
def dialogue_5(value):
    if value:
        speak(5,1)
    else :
        speak(5,2)
    return

def dialogue_6(value):
    if value:
        speak(6,1)
        speak(6,2)
        proc.kill()
    return

#def dialogue_7(value):
#    if value:
#        speak(7,1)
#    else :
#        speak(7,2)

def ending():
    time.sleep(50)
    print(" did you reach your destination ?")
    dialogue_5(True)
    if listen():
        print("enjoy your coffee, and please put me back where I initially was \n")
        print("Good bye It was nice talking to you")
        dialogue_6(True)
    else :
        print(" I see, please continue ")
        dialogue_5(False)
        ending()
    return
                    
def start():
    speak(0,1)
    if listen():
        print("Ok good, I can help you with that , but I may need your assistance, will u help me ? \n")
        dialogue_0(True)

        if listen():
            print("excellent, now take my handle at the back and say START \n")
            dialogue_1(True)

            if listen():
                print("I'm gonna start my navigation viewer now, please wait for a while \n")
                print("do you see a visualizer window with a map in it ?")
                dialogue_2(True)

                if listen():
                    print("that is great, we are almost done just a few more steps\n")
                    print(" now do you see a green path ? \n")
                    dialogue_3(True)

                    if listen():
                        print("follow the green line and it will lead you to the caffeteria \n")
                        dialogue_4(True)
                        
                        ending()

                    else :
                        print(" ok wait, i will try to display again")
                        dialogue_4(False)
                else:
                    print(" looks like I need to restart my navigation module ")
                    dialgue_3(False)
            else :
                print(" ok wait for a while it should will come up anytime")
                dialogue_2(False)
        else:
            print("Sorry but I still have to be equipped with an autonomous navigation function ")
            dialogue_1(False)

    else :
        print("ok Let me know if you ever need a coffee")
        dialogue_0(False)

                    
start()
    
    


# In[ ]:




