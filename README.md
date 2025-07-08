A Safety PPE Detection Model automates the process of monitoring compliance with PPE requirements using advanced computer vision and deep learning techniques. The model identifies whether workers are wearing the necessary protective gear by analyzing video or image data from surveillance cameras in real time. This innovation helps organizations ensure workplace safety, comply with regulations, and enhance operational efficiency. 
REQUIREMENT Libraries: 
 OpenCV: For video capture, frame resizing, and annotation. 
 ultralytics: To run YOLOv8 for object detection. 
 DeepSort: For tracking detected individuals across frames. 
Hardware/Software: 
 GPU (Optional): For faster YOLO inference during detection. 
 Python 3.x: Required to run the script. 
 Pre-trained YOLO models: 
o yolov8n.pt for person detection. 
o Custom model for safety equipment detection. 
FEATURE 1. Dual YOLO Models: 
o One model detects people in the video. 
o Another detects safety equipment (e.g., helmets, vests) and violations (e.g., no helmet). 2. Violation Tracking: 
o Logs the start and end timestamps of violations. 
o Keeps a record of individual tracking IDs associated with violations. 3. Real-time Object Tracking: 
o Tracks detected individuals across video frames using DeepSort. 4. Video Output: 
o Annotates detected objects and tracks IDs, saving the processed video to a file. 5. Logs: 
o Writes violation details to a log file for further analysis. 
