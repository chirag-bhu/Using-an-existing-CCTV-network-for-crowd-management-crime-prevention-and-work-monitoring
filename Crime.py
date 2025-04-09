# import cv2
# import numpy as np
# from datetime import datetime
# import time
# import os
# from collections import deque

# # Constants (adjusted for CPU performance)
# CONFIDENCE_THRESHOLD = 0.5
# NMS_THRESHOLD = 0.4  # Slightly higher to reduce processing load
# LOITERING_TIME_THRESHOLD = 30  # seconds
# AGGRESSIVE_MOVEMENT_THRESHOLD = 0.6  # Higher threshold to reduce false positives
# ALERT_COOLDOWN = 60  # seconds between alerts
# FRAME_SKIP = 2  # Process every nth frame to reduce CPU load

# # Load YOLO model (CPU-only version)
# def load_yolo_model():
#     weights_path = "yolov3.weights"
#     config_path = "yolov3.cfg"
    
#     if not os.path.exists(weights_path):
#         raise FileNotFoundError("YOLO weights file not found. Please download yolov3.weights.")
    
#     net = cv2.dnn.readNet(weights_path, config_path)
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
#     with open("coco.names", "r") as f:
#         classes = [line.strip() for line in f.readlines()]
    
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
#     return net, classes, output_layers

# # Weapons and suspicious items
# WEAPON_CLASSES = ["knife", "gun", "pistol", "rifle"]
# SUSPICIOUS_ITEMS = ["backpack", "suitcase", "bag"]
# VEHICLE_CLASSES = ["car", "truck", "motorcycle", "bus"]

# # Tracking systems
# tracked_objects = {}
# loitering_alerts = set()
# last_alert_time = {}

# # Background subtractor
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# def detect_suspicious_activity(frame, detections, classes, frame_count, fps):
#     alerts = []
#     current_time = time.time()
    
#     # Process each detection
#     for detection in detections:
#         class_id = detection['class_id']
#         label = classes[class_id]
#         confidence = detection['confidence']
#         box = detection['box']
        
#         # Weapon detection
#         if label in WEAPON_CLASSES and confidence > 0.7:
#             alerts.append({
#                 "type": "weapon_detected",
#                 "label": label,
#                 "confidence": confidence,
#                 "box": box,
#                 "timestamp": current_time
#             })
        
#         # Abandoned item detection
#         if label in SUSPICIOUS_ITEMS:
#             obj_id = f"{label}_{box[0]}_{box[1]}"
            
#             if obj_id not in tracked_objects:
#                 tracked_objects[obj_id] = {
#                     "first_seen": current_time,
#                     "last_seen": current_time,
#                     "box": box,
#                     "label": label
#                 }
#             else:
#                 tracked_objects[obj_id]["last_seen"] = current_time
                
#                 duration = current_time - tracked_objects[obj_id]["first_seen"]
#                 if duration > LOITERING_TIME_THRESHOLD and obj_id not in loitering_alerts:
#                     alerts.append({
#                         "type": "abandoned_item",
#                         "label": label,
#                         "duration": duration,
#                         "box": box,
#                         "timestamp": current_time
#                     })
#                     loitering_alerts.add(obj_id)
    
#     # Motion detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     fgmask = fgbg.apply(gray)
#     motion_level = np.count_nonzero(fgmask) / (frame.shape[0] * frame.shape[1])
    
#     if motion_level > AGGRESSIVE_MOVEMENT_THRESHOLD:
#         if "aggressive_movement" not in last_alert_time or \
#            (current_time - last_alert_time["aggressive_movement"] > ALERT_COOLDOWN):
#             alerts.append({
#                 "type": "aggressive_movement",
#                 "level": motion_level,
#                 "timestamp": current_time
#             })
#             last_alert_time["aggressive_movement"] = current_time
    
#     return alerts

# def draw_detections(frame, detections, alerts):
#     for detection in detections:
#         label = detection['label']
#         confidence = detection['confidence']
#         box = detection['box']
        
#         color = (0, 255, 0)  # green
#         if label in WEAPON_CLASSES:
#             color = (0, 0, 255)  # red
#         elif label in SUSPICIOUS_ITEMS:
#             color = (0, 165, 255)  # orange
        
#         cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
#         cv2.putText(frame, f"{label} {confidence:.2f}", (box[0], box[1]-5), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     for alert in alerts:
#         if alert['type'] == 'abandoned_item':
#             box = alert['box']
#             cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 3)
#             cv2.putText(frame, f"ABANDONED {alert['label']} {alert['duration']:.1f}s", 
#                        (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         if alert['type'] == 'aggressive_movement':
#             cv2.putText(frame, f"AGGRESSIVE MOVEMENT DETECTED", 
#                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
#     return frame

# def send_alert(alert):
#     timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    
#     if alert['type'] == 'weapon_detected':
#         print(f"[ALERT {timestamp}] Weapon detected: {alert['label']} (confidence: {alert['confidence']:.2f})")
#     elif alert['type'] == 'abandoned_item':
#         print(f"[ALERT {timestamp}] Abandoned item: {alert['label']} stationary for {alert['duration']:.1f} seconds")
#     elif alert['type'] == 'aggressive_movement':
#         print(f"[ALERT {timestamp}] Aggressive movement detected (level: {alert['level']:.2f})")

# def main():
#     # Load YOLO model
#     net, classes, output_layers = load_yolo_model()
    
#     # Open video source
#     cap = cv2.VideoCapture(1)
#     if not cap.isOpened():
#         print("Error: Could not open video source")
#         return
    
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Skip frames to reduce processing load
#         if frame_count % FRAME_SKIP != 0:
#             frame_count += 1
#             continue
        
#         height, width = frame.shape[:2]
        
#         # Prepare frame for YOLO (smaller size for better CPU performance)
#         blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
#         net.setInput(blob)
#         outs = net.forward(output_layers)
        
#         # Process detections
#         class_ids = []
#         confidences = []
#         boxes = []
        
#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
                
#                 if confidence > CONFIDENCE_THRESHOLD:
#                     box = detection[0:4] * np.array([width, height, width, height])
#                     (center_x, center_y, box_width, box_height) = box.astype("int")
#                     x = int(center_x - (box_width / 2))
#                     y = int(center_y - (box_height / 2))
                    
#                     boxes.append([x, y, int(box_width), int(box_height)])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)
        
#         # Apply non-max suppression
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
#         detections = []
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 detections.append({
#                     'class_id': class_ids[i],
#                     'label': classes[class_ids[i]],
#                     'confidence': confidences[i],
#                     'box': boxes[i]
#                 })
        
#         # Detect suspicious activities
#         alerts = detect_suspicious_activity(frame, detections, classes, frame_count, cap.get(cv2.CAP_PROP_FPS))
        
#         # Send alerts
#         for alert in alerts:
#             send_alert(alert)
        
#         # Draw detections and display
#         frame = draw_detections(frame, detections, alerts)
#         cv2.imshow("AI Surveillance System (CPU)", frame)
        
#         frame_count += 1
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from datetime import datetime
import time
import os
import argparse
from collections import defaultdict

# Enhanced Configuration
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
LOITERING_TIME_THRESHOLD = 30
AGGRESSIVE_MOVEMENT_THRESHOLD = 0.3  # Lower threshold for better fight detection
ALERT_COOLDOWN = 30
FRAME_SKIP = 1  # Process every frame for better fight detection

# Load YOLO model (now using YOLOv4-tiny for better performance)
def load_yolo_model():
    weights_path = "yolov4-tiny.weights"
    config_path = "yolov4-tiny.cfg"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError("YOLO weights file not found. Please download yolov4-tiny.weights.")
    
    net = cv2.dnn.readNet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

# Enhanced weapon and person detection
WEAPON_CLASSES = ["knife", "gun", "pistol", "rifle"]
PERSON_CLASS = "person"
SUSPICIOUS_ITEMS = ["backpack", "suitcase", "bag"]

def detect_fights(frame, prev_frame, motion_history, fgbg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Initialize previous frame if needed
    if prev_frame is None:
        return False, 0, gray, motion_history
    
    # Compute absolute difference between frames
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fight_detected = False
    motion_intensity = 0
    
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < 500:
            continue
            
        # Calculate motion intensity
        (x, y, w, h) = cv2.boundingRect(contour)
        motion_intensity += w * h
        
        # Detect rapid, erratic movements characteristic of fights
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 2.0:  # Filter out very wide/tall movements
            motion_history.append((x, y, w, h))
            
            # If we have enough history, check for fight patterns
            if len(motion_history) > 10:
                # Calculate movement variance
                dx = np.std([m[0] for m in motion_history[-10:]])
                dy = np.std([m[1] for m in motion_history[-10:]])
                
                # Fight detection heuristic
                if dx > 15 and dy > 15:  # Erratic movement threshold
                    fight_detected = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Normalize motion intensity
    motion_level = motion_intensity / (frame.shape[0] * frame.shape[1])
    
    return fight_detected, motion_level, gray, motion_history

def process_frame(frame, net, output_layers, classes, tracked_objects, 
                 loitering_alerts, last_alert_time, fgbg, prev_frame, 
                 motion_history, frame_count, fps):
    height, width = frame.shape[:2]
    
    # Prepare frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                'class_id': class_ids[i],
                'label': classes[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i]
            })
    
    # Detect suspicious activities
    alerts = []
    current_time = time.time()
    
    # Enhanced weapon detection
    for detection in detections:
        label = detection['label']
        confidence = detection['confidence']
        box = detection['box']
        
        if label in WEAPON_CLASSES and confidence > 0.7:
            if current_time - last_alert_time.get('weapon', 0) > ALERT_COOLDOWN:
                alerts.append({
                    "type": "weapon_detected",
                    "label": label,
                    "confidence": confidence,
                    "box": box,
                    "timestamp": current_time
                })
                last_alert_time['weapon'] = current_time
    
    # Fight detection
    fight_detected, motion_level, new_prev_frame, motion_history = detect_fights(
        frame, prev_frame, motion_history, fgbg
    )
    
    if fight_detected:
        if current_time - last_alert_time.get('fight', 0) > ALERT_COOLDOWN:
            alerts.append({
                "type": "fight_detected",
                "level": motion_level,
                "timestamp": current_time
            })
            last_alert_time['fight'] = current_time
    
    # Draw detections
    frame = draw_detections(frame, detections, alerts)
    
    return frame, alerts, new_prev_frame, motion_history

def draw_detections(frame, detections, alerts):
    for detection in detections:
        label = detection['label']
        confidence = detection['confidence']
        box = detection['box']
        
        color = (0, 255, 0)  # green
        if label in WEAPON_CLASSES:
            color = (0, 0, 255)  # red
        elif label in SUSPICIOUS_ITEMS:
            color = (0, 165, 255)  # orange
        
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (box[0], box[1]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    for alert in alerts:
        if alert['type'] == 'weapon_detected':
            box = alert['box']
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 3)
            cv2.putText(frame, f"WEAPON: {alert['label'].upper()}", 
                       (box[0], box[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if alert['type'] == 'fight_detected':
            cv2.putText(frame, "FIGHT DETECTED!", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return frame

def send_alert(alert):
    timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    
    if alert['type'] == 'weapon_detected':
        print(f"[ALERT {timestamp}] Weapon detected: {alert['label']} (confidence: {alert['confidence']:.2f})")
    elif alert['type'] == 'fight_detected':
        print(f"[ALERT {timestamp}] Fight detected (intensity: {alert['level']:.2f})")

def process_video(input_source, net, output_layers, classes):
    tracked_objects = {}
    loitering_alerts = set()
    last_alert_time = {}
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    prev_frame = None
    motion_history = []
    
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        return
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, alerts, prev_frame, motion_history = process_frame(
            frame, net, output_layers, classes, 
            tracked_objects, loitering_alerts, last_alert_time, 
            fgbg, prev_frame, motion_history, frame_count, fps
        )
        
        for alert in alerts:
            send_alert(alert)
        
        cv2.imshow("Enhanced Fight Detection System", processed_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Fight Detection System')
    parser.add_argument('--input', type=str, help='Input video file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    args = parser.parse_args()
    
    net, classes, output_layers = load_yolo_model()
    
    if args.input:
        process_video(args.input, net, output_layers, classes)
    else:
        process_video(args.camera, net, output_layers, classes)

if __name__ == "__main__":
    main()