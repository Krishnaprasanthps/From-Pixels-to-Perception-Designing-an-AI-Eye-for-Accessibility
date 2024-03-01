import cv2
import pyttsx3

import time
from time import sleep
 

 

  
# init function to get an engine instance for the speech synthesis 
engine = pyttsx3.init()


classNames = []
classFile = r"coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
    print(classNames[1])

configPath = r"ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = r"frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    # print(classIds, confs, bbox)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            print(className)
            engine.say(className)
            engine.runAndWait()
            
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture("video.mp4")
    #cap.set(3,800)
    #cap.set(4,600)
    #cap.set(10,70)


    while True:
        success, img = cap.read()
        if success==0:
            break
        result, objectInfo = getObjects(img,0.45,0.2)
        #print(objectInfo)
        cv2.imshow("Output",img)
        #cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
        
