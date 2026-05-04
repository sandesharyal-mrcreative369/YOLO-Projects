import cv2
import math
import glob
from ultralytics import YOLO
import cvzone



images = sorted(glob.glob("outputs/archive/train/images/*.jpg"))
model= YOLO("yolov8n.pt")

classNames = model.names
masked_images = cv2.imread("mask.png")

while True:
    for img_path in images:
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (640, 480))
        masked_images = cv2.resize(masked_images, (640, 480))
        mask_region = cv2.bitwise_and(frame, masked_images)
        result = model(mask_region,stream=True)


        for r in result:
            boxes = r.boxes
            print(boxes)
            for box in boxes:
                cls = int(box.cls[0])
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

                w,h = x2-x1,y2-y1
                cvzone.cornerRect(frame,(x1,y1,w,h),l=10,t=1)

                #confidence
                confidence = math.ceil(box.conf[0]*100)/100
                cvzone.putTextRect(frame,f"{confidence}%",(max(0,x1),max(0,y1-10)),1,1,offset=1)
                cvzone.putTextRect(frame,f"{classNames[cls]}",(max(0,x1),max(0,y1-10)),scale=2,offset= 2,thickness=1)

        cv2.imshow('Video', frame)
        cv2.imshow('Mask', mask_region)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()



cv2.destroyAllWindows()