# Person Counter using YOLOv8 + SORT Tracking

A real-time people counting system using OpenCV, YOLOv8, and SORT tracking algorithm.  
This project detects, tracks, and counts people moving **UP** and **DOWN** across defined lines in a video stream.

---

# 📌 Features

- Real-time person detection using YOLOv8
- Object tracking using SORT algorithm
- Counts people moving:
  - ⬆ UP
  - ⬇ DOWN
- Masked region detection for better accuracy
- Unique ID tracking to avoid duplicate counting
- Line crossing logic
- Lightweight and fast

---


# ⚙️ Requirements

Install dependencies:

```bash
pip install ultralytics
pip install opencv-python
pip install cvzone
pip install filterpy
pip install scikit-image
```

Or use:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

```bash
python people_counter.py
```

Press:

```bash
q
```

to quit the application.

---

# 🧠 Technologies Used

- Python
- OpenCV
- YOLOv8
- NumPy
- SORT Tracking Algorithm

---

# 🖼️ Recommended README Images

Add screenshots inside a folder:

```bash
results/
```

Example:

```bash
![Open Image](https://github.com/sandesharyal-mrcreative369/YOLO-Projects/blob/main/person-counter-project/results/first_result.png)

![Open Image](https://github.com/sandesharyal-mrcreative369/YOLO-Projects/blob/main/person-counter-project/results/second_result.png)
```

Then use in README:

```markdown
![Open Image](https://github.com/sandesharyal-mrcreative369/YOLO-Projects/blob/main/person-counter-project/results/first_result.png)
```

# Note
This video is taken from source: https://www.computervision.zone/topic/demo-videos/

https://github.com/sandesharyal-mrcreative369/YOLO-Projects/blob/main/person-counter-project/results/people.mp4



# 🚀 Future Improvements

- GPU acceleration
- Web camera live counting
- DeepSORT integration
- Crowd analytics
- Heatmap visualization
- Database logging

---

# 👨‍💻 Author

Sandesh Aryal
(Developed using YOLOv8 and OpenCV for real-time computer vision applications.)

---

# ⭐ GitHub Tips

Before uploading:

```bash
git init
git add .
git commit -m "Initial Commit"
git branch -M main
git remote add origin YOUR_REPO_LINK
git push -u origin main
```

---

# 📜 License

This project is open-source and free to use for educational purposes.