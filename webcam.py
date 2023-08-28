from ultralytics import YOLO
import cv2    #use to capture and display images and perform manipulations on them
import cvzone # to display detections- also display fancy rectangle
import math

cap=cv2.VideoCapture(0) #we can pass video location

cap.set(3,640) #width  640 1280
cap.set(4,480) #height 480 720
model=YOLO(r'../yolo_weights/yolov8l.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #len = 80 categories


while(True):
    success, img=cap.read()
    results=model(img,stream=True) #stream =True makes use of generators and hence is more efficient
    for i in results:
        boundingboxes=i.boxes
        for j in boundingboxes:
            # open cv method or cv2 method
            '''
            x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
            '''

            #cvzone method- more fancier bboxes
            x1,y1,x2,y2=j.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1

            cvzone.cornerRect(img,(x1,y1,w,h),colorC=(255,0,200))

            confidence=math.ceil((j.conf[0]*100))/100
            print(confidence)

            category=int(j.cls[0])


            cvzone.putTextRect(img,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=2,thickness=1)
    cv2.imshow('image',img)
    cv2.waitKey(1)
