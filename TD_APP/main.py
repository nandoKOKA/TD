import cv2
import numpy as np 
from gtts import gTTS
from playsound import playsound 



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
classNames = []


classFile = "C:\\Users\\Miguel Rebelo\\Desktop\\TD\\TD_APP\\Labels.txt"

with open(classFile,'rt') as f:
    classNames=[line.rstrip() for line in f]


configPath = 'C:\\Users\\Miguel Rebelo\\Desktop\TD\\TD_APP\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'C:\\Users\\Miguel Rebelo\\Desktop\TD\\TD_APP\\frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0), thickness=1)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)

    cv2.imshow("Output", img)  

    className = classNames[classId - 1]  # Obtém o nome da classe correspondente ao ID de classe
    if className not in labels:
        labels.append(className)  # Adiciona o nome da classe à lista de labels 


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
