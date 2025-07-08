import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import tkinter as tk
from tkinter import simpledialog, messagebox, Toplevel, Listbox, MULTIPLE, Button

# Load the two YOLOv8 models
person_model = YOLO("yolov8n.pt")  # Replace with your actual YOLOv8 person model
equipment_model = YOLO("C:/Users/9MIN/Downloads/best (2).pt")  # Replace with your actual YOLOv8 safety equipment model

# Define colors for specific classes from the equipment model
class_colors = {
    'vest': (0, 255, 0),           # Green
    'helmet': (0, 255, 0),         # Green
    'gloves': (0, 255, 0),         # Green
    'safety_boots': (0, 255, 0),   # Green
    'safety_glass': (0, 255, 0),   # Green
    'no_gloves': (0, 0, 255),      # Red
    'no_helmet': (0, 0, 255),      # Red
    'no_vest': (0, 0, 255),        # Red
    # Add more classes and their colors as needed
}

# Function to show the dialog box and select classes
def select_classes():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    selected_classes = []

    def on_select():
        selected_items = listbox.curselection()
        for item in selected_items:
            selected_classes.append(listbox.get(item))
        root.quit()

    dialog = Toplevel(root)
    dialog.title("Select Classes")

    listbox = Listbox(dialog, selectmode=MULTIPLE)
    listbox.pack(fill="both", expand=True)

    # Add class names to the listbox
    class_names = [
        "coverall", "earmuff", "gloves", "harness_laneyard", "helmet",
        "no_earmuff", "no_gloves", "no_helmet", "no_safety_boots",
        "no_safety_glass", "no_vest", "person", "safety_boots",
        "safety_glass", "vest"
    ]
    
    for cls in class_names:
        listbox.insert(tk.END, cls)

    Button(dialog, text="OK", command=on_select).pack()

    root.mainloop()
    return selected_classes

# Initialize the DeepSort tracker
tracker = DeepSort(max_age=5)

# Get the selected classes from the dialog box
selected_classes = select_classes()
if not selected_classes:
    messagebox.showwarning("No Classes Selected", "No classes selected. Exiting.")
    exit()

video_url = "C:/Users/9MIN/Downloads/Take Time to Take Care (Machine Safety).mp4"
output_url = "output_video.mp4"
log_file_path = "detections_log.txt"

# Initialize the video capture
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get the video's width, height, and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_url, fourcc, fps, (640, 480))

    # Open the log file for writing
    with open(log_file_path, "w") as log_file:
        # Dictionary to store detection status per track_id
        person_violation_status = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Finished processing video.")
                break

            # Get the current timestamp
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

            # Resize the frame
            resized_frame = cv2.resize(frame, (640, 480))  # Resize frame to 640x480

            # Perform inference using the person detection model
            person_results = person_model(source=resized_frame, conf=0.7)

            # Perform inference using the equipment detection model
            equipment_results = equipment_model(source=resized_frame, conf=0.7)

            # Prepare bounding boxes for the tracker
            bbs = []

            # Process person detections (but don't draw the bounding box)
            for result in person_results:
                for box in result.boxes:
                    cls = box.cls.item()
                    label = person_model.names[int(cls)]
                    confidence = box.conf.item()

                    # Get the bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Add to tracker only if it's a person detection
                    if label == "person":
                        bbs.append(([x1, y1, x2 - x1, y2 - y1], confidence, cls))

            # Process equipment detections
            current_violations = {}
            for result in equipment_results:
                for box in result.boxes:
                    cls = box.cls.item()
                    label = equipment_model.names[int(cls)]
                    if label not in selected_classes:
                        continue  # Skip classes that are not selected

                    confidence = box.conf.item()
                    color = class_colors.get(label, (0, 255, 255))  # Default to yellow if class not specified

                    # Get the bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw the bounding box and class name with confidence for equipment
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(resized_frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Track violations
                    if label in ["no_helmet", "no_vest", "no_gloves"]:
                        current_violations[cls] = label

            # Update the tracker with the person detections
            tracks = tracker.update_tracks(bbs, frame=resized_frame)

            # Log start time and detection type only once for each track_id
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                left, top, right, bottom = ltrb

                # Draw the tracking ID for the person
                cv2.rectangle(resized_frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
                cv2.putText(resized_frame, f'ID: {track_id}', (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Check for new violations and log them if not already logged
                if track_id not in person_violation_status and any(v in current_violations.values() for v in ["no_helmet", "no_vest", "no_gloves"]):
                    person_violation_status[track_id] = {"start": timestamp, "violation": current_violations}
                    violation_labels = ', '.join(current_violations.values())
                    log_file.write(f"Person ID: {track_id}, Violations: {violation_labels}, Start: {timestamp}\n")
                elif track_id in person_violation_status and all(v not in current_violations.values() for v in ["no_helmet", "no_vest", "no_gloves"]):
                    end_time = timestamp
                    violation_labels = ', '.join(person_violation_status[track_id]["violation"].values())
                    log_file.write(f"Person ID: {track_id}, Violations: {violation_labels}, Start: {person_violation_status[track_id]['start']}, End: {end_time}\n")
                    del person_violation_status[track_id]

            # Write the processed frame to the output video
            out.write(resized_frame)

            # Display the resized frame
            cv2.imshow("YOLOv8 Video", resized_frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and writer objects, and close all windows
    cap.release()
    out.release()  # Save the output video
    cv2.destroyAllWindows()

    print(f"Output video saved as {output_url}")
    print(f"Detections logged in {log_file_path}")

