import cv2

img = cv2.imread('Perfil.png')

classNames = []
classFile = '/Users/tiagocarvalho/Documents/GitHub/TD/TD_APP/Labels.txt'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

configPath = '/Users/tiagocarvalho/Documents/GitHub/TD/TD_APP/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/Users/tiagocarvalho/Documents/GitHub/TD/TD_APP/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img,confThreshold=0.5)
print(classIds,bbox)

for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(img,box,color=(0,255,0))
    cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
                 cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)

cv2.imshow("Output",img)
cv2.waitKey(0)
