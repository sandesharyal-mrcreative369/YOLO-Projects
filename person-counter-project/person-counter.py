import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import glob

model = YOLO("yolov8n.pt")

images = sorted(glob.glob("result/frames/frames/*.jpg"))
print(images[:5])
print(len(images))

for image in images:
    pic = cv2.imread(image)

    if  pic is None:
        continue

    pic = cv2.resize(pic, (640, 480))

    results = model(pic,stream=True)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()

    cv2.imshow("frame",pic)

cv2.destroyAllWindows()
