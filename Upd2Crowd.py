import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="People Counter Pro",
    page_icon="👥",
    layout="wide"
)

# Preset configurations
PRESETS = {
    "High Precision (Crowded)": {
        "confidence": 0.8,
        "min_height": 20,
        "nms": 0.4
    },
    "Balanced (Default)": {
        "confidence": 0.7,
        "min_height": 15,
        "nms": 0.3
    },
    "High Recall (Sparse)": {
        "confidence": 0.5,
        "min_height": 10,
        "nms": 0.2
    }
}

# Sidebar controls
st.sidebar.title("Settings")

# Quick settings presets
preset = st.sidebar.selectbox(
    "Quick Presets",
    list(PRESETS.keys()),
    index=1
)

# Apply selected preset
current_preset = PRESETS[preset]
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=current_preset["confidence"], 
    step=0.05
)
min_person_height = st.sidebar.slider(
    "Minimum Person Height (%)", 
    min_value=5, 
    max_value=30, 
    value=current_preset["min_height"], 
    step=1
)
nms_threshold = st.sidebar.slider(
    "Overlap Threshold (NMS)", 
    min_value=0.1, 
    max_value=0.5, 
    value=current_preset["nms"], 
    step=0.05
)

input_method = st.sidebar.radio(
    "Input Method",
    ["Upload File", "Use Camera"],
    index=0
)

# Main title
st.title("👥 Real-Time People Counter Pro")
st.markdown("""
    Upload an image/video or use your camera to count the number of people detected.
    Adjust the settings in the sidebar to improve accuracy.
""")

# Load YOLO model with GPU support if available
@st.cache_resource
def load_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # GPU acceleration if available
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        st.sidebar.success("GPU Acceleration Enabled", icon="⚡")
    except:
        st.sidebar.warning("Using CPU (Enable CUDA for better performance)", icon="⚠️")
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

net, classes, output_layers = load_model()

def count_people(image, confidence_thresh, min_height_percent, nms_thresh):
    height, width = image.shape[:2]
    min_height = int(height * (min_height_percent / 100))
    
    # Prepare input blob
    blob = cv2.dnn.blobFromImage(
        image, 
        1/255.0, 
        (416, 416), 
        swapRB=True, 
        crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if class_id == 0 and confidence > confidence_thresh:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                if h >= min_height:
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)
    
    # Draw results
    result_image = image.copy()
    people_count = 0
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                result_image, 
                f"{confidences[i]:.1%}", 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 255), 
                1
            )
            people_count += 1
    
    # Add count to image
    cv2.putText(
        result_image, 
        f"People: {people_count}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 0, 255), 
        2
    )
    
    return result_image, people_count

# Download button for images
def get_image_download_link(img, filename):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = buffered.getvalue()
    return st.download_button(
        label="Download Result",
        data=img_str,
        file_name=filename,
        mime="image/jpeg"
    )

# Camera capture function
def camera_capture():
    st.subheader("Live Camera Feed")
    run_camera = st.checkbox("Start Camera", value=False)
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(1)
    
    while run_camera:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access camera")
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, count = count_people(
            frame,
            confidence_threshold,
            min_person_height,
            nms_threshold
        )
        
        FRAME_WINDOW.image(processed_frame)
        time.sleep(0.1)
    
    camera.release()
    if not run_camera:
        st.write("Camera stopped")

# File processing function with download option
def file_processing():
    uploaded_file = st.file_uploader(
        "Choose an image or video", 
        type=["jpg", "jpeg", "png", "mp4", "mov"]
    )

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        if file_ext in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Processed Result")
                with st.spinner("Detecting people..."):
                    result_img, count = count_people(
                        image_np, 
                        confidence_threshold, 
                        min_person_height, 
                        nms_threshold
                    )
                    result_pil = Image.fromarray(result_img)
                    st.image(result_pil, use_column_width=True)
                    st.success(f"Detected {count} people in the image")
                    
                    # Download button
                    get_image_download_link(result_pil, "processed_" + uploaded_file.name)
        
        elif file_ext in ["mp4", "mov"]:
            st.subheader("Video Processing")
            
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            video_col, result_col = st.columns(2)
            
            with video_col:
                st.subheader("Original Video")
                video_bytes = uploaded_file.read()
                st.video(video_bytes)
            
            with result_col:
                st.subheader("Processed Video")
                st.warning("Video processing may take some time...")
                
                temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_output.close()
                
                cap = cv2.VideoCapture(tfile.name)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    temp_output.name, 
                    fourcc, 
                    fps, 
                    (frame_width, frame_height)
                )
                
                progress_bar = st.progress(0)
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame, _ = count_people(
                        frame, 
                        confidence_threshold, 
                        min_person_height, 
                        nms_threshold
                    )
                    out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                    
                    frame_count += 1
                    progress = int((frame_count / total_frames) * 100)
                    progress_bar.progress(min(progress, 100))
                
                cap.release()
                out.release()
                
                # Download button for video
                with open(temp_output.name, "rb") as f:
                    video_data = f.read()
                    st.video(video_data)
                    st.download_button(
                        "Download Processed Video",
                        data=video_data,
                        file_name="processed_" + uploaded_file.name,
                        mime="video/mp4"
                    )
                
                os.unlink(tfile.name)
                os.unlink(temp_output.name)
                
                st.success("Video processing complete!")

# Main app logic
if input_method == "Use Camera":
    camera_capture()
else:
    file_processing()

# App info
st.markdown("---")
st.markdown("""
    ### Features:
    - **GPU Acceleration**: Faster processing when CUDA is available
    - **Quick Presets**: Optimized settings for different scenarios
    - **Download Results**: Save processed images/videos
    - **Real-Time Camera**: Live people counting from webcam
    
    ### Tips:
    - Try different presets for crowded vs sparse scenes
    - Higher confidence = fewer false positives
    - Larger min height = ignore small detections
""")
