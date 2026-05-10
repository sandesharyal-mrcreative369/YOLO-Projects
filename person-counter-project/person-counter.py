import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math

#For tracking the object
from sort import *

model = YOLO("yolov8n.pt")

#Importing Video
video = cv2.VideoCapture("people.mp4")

#Adding masked images for focused detection part
masked_images = cv2.imread("mask.png")


# ------------------------------------------------------------
# OBJECT TRACKER
# ------------------------------------------------------------
# SORT Tracker Parameters:
# max_age      -> Maximum frames to keep lost object
# min_hits     -> Minimum detections before tracking starts
# iou_threshold-> Matching threshold between detections
tracker = Sort(max_age=15,min_hits=2,iou_threshold=0.3)



#Line Coordinates
#Format = x1, y1 ,x2 ,y2
line1 = [250,290,350,290]
line2 = [100,240,200,240]


#For counting the no.of objects
counter_up = []
counter_down = []

while True:
    success , pic = video.read()

    if  not success:
        continue

    pic = cv2.resize(pic, (640, 480))
    masked_images = cv2.resize(masked_images, (640, 480))
    masked_region = cv2.bitwise_and(pic,masked_images)

    results = model(masked_region,stream=True)

    list_detection = np.empty((0,5))   #Makes empty array list

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1

            #Drawing rectangle for detected object
            cvzone.cornerRect(pic,(x1,y1,w,h),l=5,t=2)

            # Detection confidence
            confidence = math.ceil(box.conf[0] * 100) / 100

            list_array = np.array([x1,y1,x2,y2,confidence])
            list_detection = np.vstack((list_detection,list_array))

    #Update tracker
    tracker_result = tracker.update(list_detection)
    print(tracker_result)

    for result in tracker_result:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)

        w,h = x2-x1,y2-y1

        cx,cy = int(x1+w //2) , int(y1+h //2)
        cv2.circle(pic,(cx,cy),5,(0,0,255),-1)

        

    #Drawing Lines for counting
    cv2.line(pic,(line1[0],line1[1]),(line1[2],line1[3]),color=(0,0,255),thickness=2)
    cv2.line(pic,(line2[0],line2[1]),(line2[2],line2[3]),color=(0,255,0),thickness=2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()

    cv2.imshow("frame",pic)

cv2.destroyAllWindows()
