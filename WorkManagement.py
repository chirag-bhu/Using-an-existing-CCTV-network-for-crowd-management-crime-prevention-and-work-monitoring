import cv2
import numpy as np
from datetime import datetime
import os
import pandas as pd
from collections import defaultdict

# Initialize YOLO model (using OpenCV's DNN module)
def initialize_yolo(model_path, config_path, classes_path):
    net = cv2.dnn.readNet(model_path, config_path)
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Use GPU if available
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    return net, classes

# Process each frame for detection
def process_frame(net, classes, frame, conf_threshold=0.5, nms_threshold=0.4):
    height, width = frame.shape[:2]
    
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    output_layers = net.getUnconnectedOutLayersNames()
    
    # Forward pass
    layer_outputs = net.forward(output_layers)
    
    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each output layer
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Prepare results
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            results.append((label, confidence, (x, y, w, h)))
    
    return results

# Main monitoring function
def monitor_workplace(cctv_source, net, classes):
    # Initialize data storage
    attendance_log = defaultdict(list)
    ppe_violations = []
    movement_patterns = defaultdict(list)
    
    # Open video source
    cap = cv2.VideoCapture(cctv_source)
    
    # Frame counter and skip rate (process every 5th frame for performance)
    frame_count = 0
    skip_frames = 5
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        
        # Process frame
        current_time = datetime.now()
        detections = process_frame(net, classes, frame)
        
        # Analyze detections
        people_in_frame = []
        safety_gear = {'helmet': False, 'vest': False, 'gloves': False}
        
        for (label, confidence, (x, y, w, h)) in detections:
            # Draw bounding box
            color = (0, 255, 0)  # default green
            if label == 'person':
                people_in_frame.append((x, y, w, h))
                color = (0, 0, 255)  # red for person
            elif label in ['helmet', 'vest', 'gloves']:
                safety_gear[label] = True
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Track people and check PPE
        for person in people_in_frame:
            # Simple tracking - in real app use proper tracking algorithm
            person_id = hash(person)  # simplistic approach - use proper ID generation
            
            # Log movement
            movement_patterns[person_id].append({
                'timestamp': current_time,
                'position': (person[0], person[1])
            })
            
            # Check for PPE violations (industrial setting)
            if not all(safety_gear.values()):
                violation = {
                    'timestamp': current_time,
                    'person_id': person_id,
                    'missing_gear': [k for k, v in safety_gear.items() if not v]
                }
                ppe_violations.append(violation)
                cv2.putText(frame, "PPE VIOLATION!", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Work Monitoring', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Generate reports
    generate_reports(attendance_log, ppe_violations, movement_patterns)

# Generate reports from collected data
def generate_reports(attendance_log, ppe_violations, movement_patterns):
    # Attendance report
    attendance_df = pd.DataFrame([
        {'employee_id': emp_id, 'time_in': min(times), 'time_out': max(times)}
        for emp_id, times in attendance_log.items()
    ])
    
    # PPE violations report
    ppe_df = pd.DataFrame(ppe_violations)
    
    # Movement analysis (heatmap could be generated from this)
    movement_data = []
    for emp_id, positions in movement_patterns.items():
        for pos in positions:
            movement_data.append({
                'employee_id': emp_id,
                'timestamp': pos['timestamp'],
                'x_position': pos['position'][0],
                'y_position': pos['position'][1]
            })
    movement_df = pd.DataFrame(movement_data)
    
    # Save reports
    os.makedirs('reports', exist_ok=True)
    attendance_df.to_csv('reports/attendance.csv', index=False)
    ppe_df.to_csv('reports/ppe_violations.csv', index=False)
    movement_df.to_csv('reports/movement_patterns.csv', index=False)
    
    print("Reports generated in 'reports' directory")

# Main function
def main():
    # Paths to YOLO files (download from official YOLO website)
    model_path = 'yolov3.weights'
    config_path = 'yolov3.cfg'
    classes_path = 'coco.names'
    
    # CCTV source (0 for webcam, or path to video file)
    cctv_source = 1  # or 'path/to/your/cctv_feed.mp4'
    
    # Initialize YOLO
    net, classes = initialize_yolo(model_path, config_path, classes_path)
    
    # Start monitoring
    monitor_workplace(cctv_source, net, classes)

if __name__ == '__main__':
    main()