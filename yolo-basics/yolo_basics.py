from ultralytics import YOLO
import cv2

# ---------------------------------------------------------
# YOLO Object Detection using a Pretrained YOLOv8 Model
# ---------------------------------------------------------
# This script loads a pretrained YOLOv8 model and performs
# object detection on an input image.
#
# The detected objects are automatically saved as a new image
# with bounding boxes and labels.
# ---------------------------------------------------------



# Load pretrained YOLO model
# Make sure the model file path is correct
# Autmatically downloads if not present yolov8n.pt
model = YOLO("yoloStart/yolov8n.pt")

# Run object detection on the input image
# save=True will store the output image in the "runs/detect/" folder
results = model("bus-detected.jpg",save=True)

# Close all OpenCV windows after key press
cv2.waitKey(0)
