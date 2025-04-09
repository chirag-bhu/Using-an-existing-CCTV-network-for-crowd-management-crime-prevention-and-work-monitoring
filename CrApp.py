import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
import os
from collections import defaultdict
import tempfile

# Page configuration
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🛡️",
    layout="wide"
)

# Sidebar controls
st.sidebar.title("Settings")

# Confidence thresholds
st.sidebar.subheader("Detection Thresholds")
weapon_threshold = st.sidebar.slider("Weapon Confidence", 0.0, 1.0, 0.7, 0.01)
fight_threshold = st.sidebar.slider("Fight Sensitivity", 0.0, 1.0, 0.3, 0.01)
item_threshold = st.sidebar.slider("Suspicious Item Confidence", 0.0, 1.0, 0.5, 0.01)

# Alert settings
st.sidebar.subheader("Alert Settings")
alert_cooldown = st.sidebar.slider("Alert Cooldown (seconds)", 1, 120, 30, 1)
frame_skip = st.sidebar.slider("Frame Skip", 1, 10, 2, 1)

# Constants
NMS_THRESHOLD = 0.4
LOITERING_TIME_THRESHOLD = 30
FRAME_SKIP = frame_skip

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    weights_path = "yolov4-tiny.weights"
    config_path = "yolov4-tiny.cfg"
    
    if not os.path.exists(weights_path):
        st.error("YOLO weights file not found. Please download yolov4-tiny.weights.")
        return None, None, None
    
    net = cv2.dnn.readNet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

net, classes, output_layers = load_yolo_model()

# Detection classes
WEAPON_CLASSES = ["knife", "gun", "pistol", "rifle"]
PERSON_CLASS = "person"
SUSPICIOUS_ITEMS = ["backpack", "suitcase", "bag"]

# Initialize session state
if 'tracked_objects' not in st.session_state:
    st.session_state.tracked_objects = {}
if 'loitering_alerts' not in st.session_state:
    st.session_state.loitering_alerts = set()
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = st.empty()

# Main app
st.title("AI Surveillance System 🛡️")
st.write("Detect weapons, fights, and suspicious items in real-time")

# Input selection
input_option = st.radio("Select input source:", 
                       ["Webcam", "Upload Image", "Upload Video"])

def detect_fights(frame, prev_frame, motion_history):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if prev_frame is None:
        return False, 0, gray, motion_history
    
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fight_detected = False
    motion_intensity = 0
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
            
        (x, y, w, h) = cv2.boundingRect(contour)
        motion_intensity += w * h
        
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 2.0:
            motion_history.append((x, y, w, h))
            
            if len(motion_history) > 10:
                dx = np.std([m[0] for m in motion_history[-10:]])
                dy = np.std([m[1] for m in motion_history[-10:]])
                
                if dx > 15 and dy > 15:
                    fight_detected = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    motion_level = motion_intensity / (frame.shape[0] * frame.shape[1])
    
    return fight_detected, motion_level, gray, motion_history

def process_frame(frame, net, output_layers, classes, frame_count):
    height, width = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > item_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, item_threshold, NMS_THRESHOLD)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                'class_id': class_ids[i],
                'label': classes[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i]
            })
    
    return detections

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
    alert_text = ""
    
    if alert['type'] == 'weapon_detected':
        alert_text = f"[ALERT {timestamp}] Weapon detected: {alert['label']} (confidence: {alert['confidence']:.2f})"
    elif alert['type'] == 'fight_detected':
        alert_text = f"[ALERT {timestamp}] Fight detected (intensity: {alert['level']:.2f})"
    
    st.session_state.alerts.append(alert_text)
    if len(st.session_state.alerts) > 10:
        st.session_state.alerts.pop(0)

# Process webcam
if input_option == "Webcam":
    run_webcam = st.checkbox("Start Webcam")
    
    if run_webcam and net is not None:
        cap = cv2.VideoCapture(0)
        prev_frame = None
        motion_history = []
        frame_count = 0
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            if frame_count % FRAME_SKIP != 0:
                frame_count += 1
                continue
            
            # Process frame
            detections = process_frame(frame, net, output_layers, classes, frame_count)
            
            # Detect suspicious activities
            current_time = time.time()
            alerts = []
            
            # Weapon detection
            for detection in detections:
                label = detection['label']
                confidence = detection['confidence']
                box = detection['box']
                
                if label in WEAPON_CLASSES and confidence > weapon_threshold:
                    if current_time - st.session_state.last_alert_time.get('weapon', 0) > alert_cooldown:
                        alerts.append({
                            "type": "weapon_detected",
                            "label": label,
                            "confidence": confidence,
                            "box": box,
                            "timestamp": current_time
                        })
                        st.session_state.last_alert_time['weapon'] = current_time
            
            # Fight detection
            fight_detected, motion_level, prev_frame, motion_history = detect_fights(
                frame, prev_frame, motion_history
            )
            
            if fight_detected and motion_level > fight_threshold:
                if current_time - st.session_state.last_alert_time.get('fight', 0) > alert_cooldown:
                    alerts.append({
                        "type": "fight_detected",
                        "level": motion_level,
                        "timestamp": current_time
                    })
                    st.session_state.last_alert_time['fight'] = current_time
            
            # Send alerts
            for alert in alerts:
                send_alert(alert)
            
            # Draw detections
            frame = draw_detections(frame, detections, alerts)
            
            # Display frame
            st.session_state.frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            frame_count += 1
            
            if st.button("Stop Webcam"):
                run_webcam = False
                cap.release()
        
        cap.release()

# Process uploaded image
elif input_option == "Upload Image" and net is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process frame
        detections = process_frame(frame, net, output_layers, classes, 0)
        
        # Weapon detection
        current_time = time.time()
        alerts = []
        
        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            box = detection['box']
            
            if label in WEAPON_CLASSES and confidence > weapon_threshold:
                alerts.append({
                    "type": "weapon_detected",
                    "label": label,
                    "confidence": confidence,
                    "box": box,
                    "timestamp": current_time
                })
        
        # Draw detections
        frame = draw_detections(frame, detections, alerts)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, channels="BGR", caption="Processed Image", use_column_width=True)
        with col2:
            if alerts:
                st.warning("Alerts Detected:")
                for alert in alerts:
                    if alert['type'] == 'weapon_detected':
                        st.error(f"Weapon detected: {alert['label']} (confidence: {alert['confidence']:.2f})")
            else:
                st.success("No threats detected")

# Process uploaded video
elif input_option == "Upload Video" and net is not None:
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        prev_frame = None
        motion_history = []
        frame_count = 0
        
        stframe = st.empty()
        stop_button = st.button("Stop Video Processing")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % FRAME_SKIP != 0:
                frame_count += 1
                continue
            
            # Process frame
            detections = process_frame(frame, net, output_layers, classes, frame_count)
            
            # Detect suspicious activities
            current_time = time.time()
            alerts = []
            
            # Weapon detection
            for detection in detections:
                label = detection['label']
                confidence = detection['confidence']
                box = detection['box']
                
                if label in WEAPON_CLASSES and confidence > weapon_threshold:
                    if current_time - st.session_state.last_alert_time.get('weapon', 0) > alert_cooldown:
                        alerts.append({
                            "type": "weapon_detected",
                            "label": label,
                            "confidence": confidence,
                            "box": box,
                            "timestamp": current_time
                        })
                        st.session_state.last_alert_time['weapon'] = current_time
            
            # Fight detection
            fight_detected, motion_level, prev_frame, motion_history = detect_fights(
                frame, prev_frame, motion_history
            )
            
            if fight_detected and motion_level > fight_threshold:
                if current_time - st.session_state.last_alert_time.get('fight', 0) > alert_cooldown:
                    alerts.append({
                        "type": "fight_detected",
                        "level": motion_level,
                        "timestamp": current_time
                    })
                    st.session_state.last_alert_time['fight'] = current_time
            
            # Send alerts
            for alert in alerts:
                send_alert(alert)
            
            # Draw detections
            frame = draw_detections(frame, detections, alerts)
            
            # Display frame
            stframe.image(frame, channels="BGR", use_column_width=True)
            
            frame_count += 1
        
        cap.release()
        os.unlink(tfile.name)

# Display alerts
if st.session_state.alerts:
    st.sidebar.subheader("Recent Alerts")
    for alert in reversed(st.session_state.alerts[-5:]):
        st.sidebar.warning(alert)
else:
    st.sidebar.info("No alerts detected yet")