import cv2
import numpy as np 
from gtts import gTTS
from playsound import playsound 
import cvlib as cv # install the required packages (numpy, opencv-python, requests, progressbar, pillow, tensorflow, keras)
from cvlib.object_detection import draw_bbox


#funcao que diz os objectos que foram vistos
def fala (texto):
    print(texto)
    language= "en"
    output= gTTS(text=texto, lang=language, slow=False) 

    output.save("C:\\Users\\Miguel Rebelo\\Desktop\\TD\\TD_APP\\sons\\output.mp3")
    playsound("C:\\Users\\Miguel Rebelo\\Desktop\\TD\\TD_APP\\sons\\output.mp3")

thres=0.5
#img = cv2.imread('C:\\Users\\Miguel Rebelo\\Desktop\\TD\\TD_APP\\Perfil.png')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

labels=[]


while True:
    ret, frame = cap.read()
    bbox, label, conf = cv.detect_common_objects(frame)
    output_image= draw_bbox(frame, bbox, label, conf)

    cv2.imshow("Object Detection", output_image)

    for item in label:
        if item in labels:
            pass
        else:
            labels.append(item)
            
    if cv2.waitKey(1) & 0xFF == ord("q") :
        break 


i = 0
frase = []
for label in labels:
    if i == 0:
        frase.append(f"I found a: {label}")
    else:
        frase.append(f"a {label}")
    i += 1

if len(frase) > 1:
    last_item = frase.pop()  # Remove o último item da lista
    frase.append(f"and {last_item}")

fala(", ".join(frase))
