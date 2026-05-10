import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import glob

model = YOLO("yolov8n.pt")

# images = sorted(glob.glob("result/frames/frames/*.jpg"))
# print(images[:5])
# print(len(images))



# for image in images:
#     pic = cv2.imread(image)


video = cv2.VideoCapture("people.mp4")

while True:
    success , pic = video.read()

    if  not success:
        continue

    pic = cv2.resize(pic, (640, 480))
    results = model(pic,stream=True)


for r in results:
    boxes = r.boxes

    for box in boxes:
        x1,y1,x2,y2 = box.xyxy[0]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

        cvzone.cornerRect(pic,(x1,y1,x2,y2))

    if cv2.waitKey(30) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()

    cv2.imshow("frame",pic)

cv2.destroyAllWindows()
