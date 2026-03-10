****Blind Assistance with Real-Time Object Detection System****

**Project Overview**

This project presents a computer vision-based solution to assist visually impaired individuals. By utilizing a Deep Convolutional Neural Network (YOLOv8), the system identifies everyday objects and provides spatial audio feedback. The system is optimized for real-time performance on standard hardware using multi-threading to ensure zero-lag audio-visual synchronization

**Tech Stacks**

--Language: Python 3.10 (The core engine).
--AI Framework: YOLOv8 by Ultralytics (A Deep CNN framework that identifies objects).
--Computer Vision Framework: OpenCV (Handles the camera feed and visual processing).
--Audio Engine: pyttsx3 (Offline Text-to-Speech library).
--Concurrency: Threading & Queue (Python’s built-in libraries for "No-Lag" performance).

**Features & Advantages**

1. Object Recognition: Identifies 80 different classes (People, cars, chairs, etc.) using the COCO dataset.
2. Spatial Guidance: Tells the user if an object is Left, Front, or Right.
3. No-Lag Audio: Uses Multi-threading so the voice doesn't pause the video feed.
4. Privacy Focused: Works 100% Offline—no data ever leaves the device.
5. Smart Filtering: Uses a 3-second cooldown to prevent the same object from being announced repeatedly.
6. Speed: YOLOv8 is a "Single-Shot" detector, making it fast enough for real-time use on a standard laptop CPU.
7. Reliability: Because it is offline, it works in basements, rural areas, or elevators where there is no internet.
8. Low Cognitive Load: The system summarizes objects (e.g., "2 chairs front") instead of shouting every single detection.
9. Scalable: Easy to add new features like face recognition or distance estimation in the future.
