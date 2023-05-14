import cv2
import numpy as np 
from gtts import gTTS
from playsound import playsound
import math

# Função que diz os objetos que foram vistos
def fala(texto):
    print(texto)
    language = "en"
    output = gTTS(text=texto, lang=language, slow=False) 
    output.save("/Users/tiagocarvalho/Documents/GitHub/TD/TD_APP/sons/output.mp3")
    playsound("/Users/tiagocarvalho/Documents/GitHub/TD/TD_APP/sons/output.mp3")

thres = 0.5
#url="http://192.168.1.5:8080/video"
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

labels = []
classNames = []
detections =[]

classFile = "/Users/tiagocarvalho/Documents/GitHub/TD/TD_APP/Labels.txt"

with open(classFile, 'rt') as f:
    classNames = [line.rstrip() for line in f]

configPath = '/Users/tiagocarvalho/Documents/GitHub/TD/TD_APP/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/Users/tiagocarvalho/Documents/GitHub/TD/TD_APP/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def get_cardinal_direction(angle):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    index = round(angle / (360. / len(directions)))
    return directions[index]

coordenadas_exibicao = []  # Move a definição da lista para fora do loop

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds,bbox) #mostra as leituras

    # Limpa as coordenadas exibidas anteriormente
    coordenadas_exibicao.clear()  # Limpa a lista a cada iteração

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
            if classId - 1 < len(classNames):
                className = classNames[classId - 1].upper()
                cv2.putText(img, className, (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                if className not in labels:
                    labels.append(className)

                # Armazena as coordenadas da caixa delimitadora
                coordenadas_exibicao.append((className, (box[0]+10, box[1])))
                                            
     # Exibe as coordenadas e pontos cardeais dos objetos detectados
    for classe, (x, y) in coordenadas_exibicao:
        text = f"{classe}: ({x}, {y})"
        
        # Calcula o ângulo em relação ao centro da imagem
        img_center_x = img.shape[1] // 2
        img_center_y = img.shape

        # Calcula o ângulo em relação ao centro da imagem
        img_center_x = img.shape[1] // 2
        img_center_y = img.shape[0] // 2
        dx = x - img_center_x
        dy = img_center_y - y
        angle = math.atan2(dy, dx) * 180 / math.pi

        # Obtém o ponto cardeal correspondente ao ângulo
        cardinal_direction = get_cardinal_direction(angle)
        text += f" ({cardinal_direction})"

            # Exibe o texto com as coordenadas e pontos cardeais
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Output", img)  

    if cv2.waitKey(1) & 0xFF == ord("q"):
         break

frase = []
unique_classes = set()

for classe, (x, y) in coordenadas_exibicao:
    if classe not in unique_classes:
        unique_classes.add(classe)

        if classe not in labels:
            labels.append(classe)

    frase.append(f"a {classe} at ({x}, {y}) {cardinal_direction}")

if len(frase) > 1:
    last_item = frase.pop() # Remove o último item da lista
    frase.append(f"and {last_item}")

fala(", ".join(frase))

cv2.destroyAllWindows()
cap.release()

