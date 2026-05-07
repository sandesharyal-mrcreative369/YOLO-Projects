import cv2  # Import OpenCV for image processing and visualization
import math  # Import math module for mathematical operations

# Import glob for reading multiple image files from a folder
import glob

# Import YOLO model from Ultralytics
from ultralytics import YOLO

# Import cvzone for better UI elements and bounding boxes
import cvzone

# Import SORT tracking algorithm
from sort import *

import numpy as np  # Import NumPy for array handling


# LOAD IMAGES
# ------------------------------------------------------------
# Reads all JPG images from the dataset folder
# sorted() ensures images are processed in correct sequence

images = sorted(glob.glob("outputs/archive/train/images/*.jpg"))

#LOAD YOLO MODEL
# ------------------------------------------------------------
# Loads pretrained YOLOv8 nano model
# You can replace with custom trained model if needed
model= YOLO("yolov8n.pt")


classNames = model.names   # Stores all class names from YOLO model


#LOAD MASK IMAGE
# ------------------------------------------------------------
# Mask image is used to focus detection only on specific region
masked_images = cv2.imread("mask.png")

# ------------------------------------------------------------
# OBJECT TRACKER
# ------------------------------------------------------------
# SORT Tracker Parameters:
# max_age      -> Maximum frames to keep lost object
# min_hits     -> Minimum detections before tracking starts
# iou_threshold-> Matching threshold between detections
tracker = Sort(max_age=15,min_hits=2,iou_threshold=0.3)

#line coordinates
# format: x1  y1  x2  y2
line1 = [140,145,250,145]

# Stores unique IDs of counted vehicles
# Prevents duplicate counting
total_count= []


#Main Loop
# Processes each image frame one by one
for img_path in images:
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (640, 480))
    masked_images = cv2.resize(masked_images, (640, 480))


    #------------------------------------------------------------
    #APPLY MASK
    # ------------------------------------------------------------
    # Keeps only masked region visible
    # Useful for reducing unwanted detections
    mask_region = cv2.bitwise_and(frame, masked_images)

    # ------------------------------------------------------------
    # YOLO DETECTION
    # ------------------------------------------------------------
    # stream=True returns generator for faster processing
    result = model(mask_region,stream=True)


    # ------------------------------------------------------------
    # EMPTY DETECTION ARRAY
    # ------------------------------------------------------------
    # Format required by SORT tracker:
    # [x1, y1, x2, y2, confidence]
    list_detection = np.empty((0, 5))  #Makes empty array list


    # PROCESS YOLO RESULTS
    for r in result:
        boxes = r.boxes

        for box in boxes:

            x1,y1,x2,y2 = box.xyxy[0]      # Get bounding box coordinates
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)  #Convert float -> int

            w,h = x2-x1,y2-y1      # WIDTH AND HEIGHT

            # DRAW CORNER RECTANGLE
            cvzone.cornerRect(frame,(x1,y1,w,h),l=10,t=1)

            #Detection confidence
            confidence = math.ceil(box.conf[0]*100)/100

            #cvzone.putTextRect(frame,f"{confidence}%",(max(0,x1),max(0,y1-10)),1,1,offset=1)

            #Drawing line for counting
            cv2.line(frame,(line1[0],line1[1]),(line1[2],line1[3]),(255,0,0),2)

            #Get className
            cls = int(box.cls[0])
            class_classified = classNames[cls]

            # VEHICLE FILTERING
            # Only detect vehicles with confidence greater than 0.4
            if class_classified in ["car", "bus", "truck"] and confidence > 0.4:

                # Show detected class name
                #cvzone.putTextRect(frame,f"{class_classified}",(max(0,x1),max(0,y1-10)),scale=2,offset= 2,thickness=1)

                # STORE DETECTIONS FOR TRACKER
                list_array = np.array([x1,y1,x2,y2,confidence])

                # Add detection into tracker list
                list_detection = np.vstack((list_detection,list_array))

    # UPDATE TRACKER
    tracker_results = tracker.update(list_detection)

    # TRACK EACH VEHICLE
    # For getting IDs too.
    for trackers in tracker_results:

        x1,y1,x2,y2,ID = trackers
        x1,y1,x2,y2,ID = int(x1),int(y1),int(x2),int(y2),int(ID)
        print(ID)
        w,h = x2-x1,y2-y1
        cx, cy = int(x1 + w // 2), int(y1 + h // 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), thickness=-1)
        cvzone.putTextRect(frame,f"{ID}",(max(0,x1),max(0,y1-10)),scale=2,offset= 2,thickness=1)

        # VEHICLE COUNTING LOGIC
        # Checks whether vehicle crosses counting line
        if line1[0] < cx < line1[2] and line1[1] - 30 < cy < line1[3] + 30:

            # Prevent duplicate counting
            if total_count.count(ID) == 0:

                # Store unique ID
                total_count.append(ID)

    # DISPLAY TOTAL COUNT
    cvzone.putTextRect(frame, f"Count: {len(total_count)}", (30, 40), scale=2, offset=2, thickness=1)

    # SHOW OUTPUT WINDOWS
    cv2.imshow('Video', frame)
    cv2.imshow('Mask', mask_region)

    # EXIT CONDITION
    # Press Q to quit program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()



# Close all windows after loop ends
cv2.destroyAllWindows()