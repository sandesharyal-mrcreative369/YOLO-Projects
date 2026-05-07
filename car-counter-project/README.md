## Project Title

YOLO Vehicle Detection, Tracking, and Counting System

---

## Overview

This project performs real-time vehicle detection, tracking, and counting using YOLOv8, OpenCV, SORT Tracking Algorithm, and CvZone.

The system detects vehicles such as:

* Cars
* Buses
* Trucks

Each detected vehicle is assigned a unique ID using SORT tracking. Vehicles are counted when they cross a predefined line.

---

## Features

* Real-time vehicle detection using YOLOv8
* Vehicle tracking with SORT algorithm
* Unique ID assignment for each vehicle
* Vehicle counting system
* Region masking support
* Bounding boxes and center point visualization
* Confidence filtering
* Counting line visualization

---

## Technologies Used

* Python
* OpenCV
* YOLOv8
* CvZone
* NumPy
* SORT Tracking Algorithm

---

## Project Structure

```text
project/
│
├── main.py
├── mask.png
├── yolov8n.pt
├── sort.py
│
├── outputs/
│   └── archive/
│       └── train/
│           └── images/
│               ├── 1.jpg
│               ├── 2.jpg
│               ├── ...
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/sandesharyal-mrcreative369
```

### Install Required Libraries

```bash
pip install ultralytics
pip install opencv-python
pip install cvzone
pip install numpy
```

---

## How It Works

1. Images are loaded from the dataset folder.
2. A mask is applied to focus detection only on specific region.
3. YOLOv8 detects vehicles.
4. SORT tracker assigns unique IDs.
5. Vehicles crossing the counting line are counted.
6. Total vehicle count is displayed on screen.

---

## Vehicle Counting Logic

Vehicles are counted only once.

The program checks:

* Whether the vehicle center crosses the counting line
* Whether the vehicle ID already exists in count list

This prevents duplicate counting.

---

## Output

The program displays:

* Bounding boxes
* Vehicle class names
* Tracking IDs
* Vehicle count
* Counting line
* Masked detection area

---

## Future Improvements

* Real-time webcam support
* Video file support
* Speed estimation
* Lane detection
* Traffic analysis dashboard
* Custom trained YOLO model

---

## Author

Developed using YOLOv8, OpenCV, and SORT Tracking Algorithm.
