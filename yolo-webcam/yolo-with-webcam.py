# ============================================================
# YOLOv8 Real-Time Object Detection using Webcam
# ------------------------------------------------------------
# This project uses a webcam feed and the YOLOv8 model
# to detect objects in real time.
#
# Libraries Used:
# - ultralytics : for YOLOv8 object detection
# - cv2         : for webcam capture and image display
# - cvzone      : for enhanced UI elements like corner boxes
# - math        : for rounding confidence values
#
# Controls:
# - Press 'q' to quit the application
# ============================================================


from ultralytics import YOLO
import cv2
import cvzone
import math

# Open webcam (0 = default webcam)
video = cv2.VideoCapture(0)

# Load pretrained YOLOv8 nano model
# Make sure "yolov8n.pt" is in the same folder as this script
model = YOLO("yolov8n.pt")

# Load built-in class names from YOLO model
classNames =model.names

# Start infinite loop for continuous webcam detection
while True:

    success , image = video.read()   # Read a frame from webcam
    image = cv2.flip(image,1)   # Flip image horizontally for mirror effect
    results = model(image,stream=True)    # Perform object detection on current frame

    if not success:
        break

    # Process detection results
    for r in results:
        boxes = r.boxes

        # Loop through each detected object
        for box in boxes:

             x1,y1,x2,y2 = box.xyxy[0]      # Get bounding box coordinates
             x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)     # Convert coordinates into integers

             # Draw rectangle using openCV
            # cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)

             w,h = x2-x1,y2-y1     # Calculate width and height of bounding box
             cvzone.cornerRect(image,(x1,y1,w,h))   # Draw stylish rectangle around detected object

             #For finding confidence
             confidence = math.ceil(box.conf[0]*100)/100    # Calculate confidence score

             # Display confidence score
             cvzone.putTextRect(image,f"{confidence}",(max(0,x1+15),max(0,y1-20)),scale=2,offset=3)

             #Class
             cls = int(box.cls[0])     # Get detected object class index

             # Display detected object class name
             cvzone.putTextRect(image,f"{classNames[cls]}",(max(0,x1+110),max(0,y1-20)),scale=2,offset= 2)

    # Press 'q' to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Show processed frame in a window
    cv2.imshow("Image",image)


# Release webcam resources
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
