import cv2
import math
import glob
from ultralytics import YOLO
import cvzone
from sort import *
import numpy as np

images = sorted(glob.glob("outputs/archive/train/images/*.jpg"))
model= YOLO("yolov8n.pt")

classNames = model.names
masked_images = cv2.imread("mask.png")

#Tracking the Object
tracker = Sort(max_age=15,min_hits=2,iou_threshold=0.3)

#line coordinates-->x1  y1  x2  y2
line1 = [140,145,250,145]
total_count= []

for img_path in images:
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (640, 480))
    masked_images = cv2.resize(masked_images, (640, 480))
    mask_region = cv2.bitwise_and(frame, masked_images)
    result = model(mask_region,stream=True)

    list_detection = np.empty((0, 5))  #Makes empty array list

    for r in result:
        boxes = r.boxes

        for box in boxes:

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            w,h = x2-x1,y2-y1
            cvzone.cornerRect(frame,(x1,y1,w,h),l=10,t=1)

            #confidence
            confidence = math.ceil(box.conf[0]*100)/100
            #cvzone.putTextRect(frame,f"{confidence}%",(max(0,x1),max(0,y1-10)),1,1,offset=1)

            #Drawing line
            cv2.line(frame,(line1[0],line1[1]),(line1[2],line1[3]),(255,0,0),2)

            #className
            cls = int(box.cls[0])
            class_classified = classNames[cls]

            #Checking class condition
            if class_classified == "car" or class_classified == "bus" \
            or class_classified == "truck" and confidence>0.4:

                #cvzone.putTextRect(frame,f"{class_classified}",(max(0,x1),max(0,y1-10)),scale=2,offset= 2,thickness=1)
                list_array = np.array([x1,y1,x2,y2,confidence])
                list_detection = np.vstack((list_detection,list_array))


    tracker_results = tracker.update(list_detection)

    # For getting IDs
    for trackers in tracker_results:
        x1,y1,x2,y2,ID = trackers
        x1,y1,x2,y2,ID = int(x1),int(y1),int(x2),int(y2),int(ID)
        print(ID)
        w,h = x2-x1,y2-y1
        cx, cy = int(x1 + w // 2), int(y1 + h // 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), thickness=-1)
        cvzone.putTextRect(frame,f"{ID}",(max(0,x1),max(0,y1-10)),scale=2,offset= 2,thickness=1)

        if line1[0] < cx < line1[2] and line1[1] - 30 < cy < line1[3] + 30:
            if total_count.count(ID) == 0:
                total_count.append(ID)


    cvzone.putTextRect(frame, f"Count: {len(total_count)}", (30, 40), scale=2, offset=2, thickness=1)

    cv2.imshow('Video', frame)
    cv2.imshow('Mask', mask_region)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()



cv2.destroyAllWindows()