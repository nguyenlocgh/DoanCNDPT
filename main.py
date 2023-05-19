import cv2
import random
nms_threshold = 0.2
thres = 0.55
cap = cv2.VideoCapture('https://192.168.158.62:8080/video')
cap.set(3,1280)
cap.set(4,720)
cap.set(10,150)

classNames=[]
classFile= 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames)
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath ='frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    resized = cv2.resize(img,(700,700))
    classIds, confs, bbox = net.detect(resized, confThreshold=thres)
    print(classIds,bbox)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0),
              (0, 128, 0), (0, 0, 128), (128, 128, 0)]

    if len(classIds) != 0:
        for i, (classId, confidence, box) in enumerate(zip(classIds.flatten(), confs.flatten(), bbox)):
            color = colors[i % len(colors)]
            cv2.rectangle(resized, box, color=color, thickness=2)
            cv2.putText(resized, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.putText(resized, str(round(confidence * 100)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

    cv2.imshow("Frame",resized)
    cv2.waitKey(1)