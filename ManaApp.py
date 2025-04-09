# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# import time

# # Page configuration
# st.set_page_config(
#     page_title="People Counter",
#     page_icon="👥",
#     layout="wide"
# )

# # Sidebar controls
# st.sidebar.title("Settings")
# input_method = st.sidebar.radio(
#     "Input Method",
#     ["Upload File", "Use Camera"],
#     index=0,
#     help="Choose between uploading a file or using your webcam"
# )

# confidence_threshold = st.sidebar.slider(
#     "Confidence Threshold", 
#     min_value=0.1, 
#     max_value=0.9, 
#     value=0.7, 
#     step=0.05,
#     help="Higher values reduce false positives but may miss some people"
# )
# min_person_height = st.sidebar.slider(
#     "Minimum Person Height (%)", 
#     min_value=5, 
#     max_value=30, 
#     value=15, 
#     step=1,
#     help="Filters out small detections that are likely not people"
# )
# nms_threshold = st.sidebar.slider(
#     "Overlap Threshold (NMS)", 
#     min_value=0.1, 
#     max_value=0.5, 
#     value=0.3, 
#     step=0.05,
#     help="Controls how much overlapping boxes are suppressed"
# )

# # Main title
# st.title("👥 Real-Time People Counter")
# st.markdown("""
#     Upload an image/video or use your camera to count the number of people detected.
#     Adjust the settings in the sidebar to improve accuracy.
# """)

# # Load YOLO model
# @st.cache_resource
# def load_model():
#     net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#     with open("coco.names", "r") as f:
#         classes = [line.strip() for line in f.readlines()]
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#     return net, classes, output_layers

# net, classes, output_layers = load_model()

# def count_people(image, confidence_thresh, min_height_percent, nms_thresh):
#     height, width = image.shape[:2]
#     min_height = int(height * (min_height_percent / 100))
    
#     # Prepare input blob
#     blob = cv2.dnn.blobFromImage(
#         image, 
#         1/255.0, 
#         (416, 416), 
#         swapRB=True, 
#         crop=False
#     )
#     net.setInput(blob)
#     outs = net.forward(output_layers)
    
#     # Process detections
#     boxes = []
#     confidences = []
#     class_ids = []
    
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
            
#             if class_id == 0 and confidence > confidence_thresh:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
                
#                 if h >= min_height:
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)
#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)
    
#     # Apply NMS
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)
    
#     # Draw results
#     result_image = image.copy()
#     people_count = 0
    
#     if len(indices) > 0:
#         for i in indices.flatten():
#             x, y, w, h = boxes[i]
#             cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(
#                 result_image, 
#                 f"{confidences[i]:.1%}", 
#                 (x, y - 5), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.5, 
#                 (0, 255, 255), 
#                 1
#             )
#             people_count += 1
    
#     # Add count to image
#     cv2.putText(
#         result_image, 
#         f"People: {people_count}", 
#         (20, 40), 
#         cv2.CAP_PROP_FRAME_WIDTH, 
#         1, 
#         (0, 0, 255), 
#         2
#     )
    
#     return result_image, people_count

# # Camera capture function
# def camera_capture():
#     st.subheader("Live Camera Feed")
#     run_camera = st.checkbox("Start Camera", value=False)
#     FRAME_WINDOW = st.image([])
#     camera = cv2.VideoCapture(2)
    
#     while run_camera:
#         ret, frame = camera.read()
#         if not ret:
#             st.error("Failed to access camera")
#             break
            
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         processed_frame, count = count_people(
#             frame,
#             confidence_threshold,
#             min_person_height,
#             nms_threshold
#         )
        
#         FRAME_WINDOW.image(processed_frame)
        
#         # Add small delay to prevent high CPU usage
#         time.sleep(0.1)
    
#     camera.release()
#     if not run_camera:
#         st.write("Camera stopped")

# # File processing function
# def file_processing():
#     uploaded_file = st.file_uploader(
#         "Choose an image or video", 
#         type=["jpg", "jpeg", "png", "mp4", "mov"]
#     )

#     if uploaded_file is not None:
#         file_ext = uploaded_file.name.split(".")[-1].lower()
        
#         if file_ext in ["jpg", "jpeg", "png"]:
#             image = Image.open(uploaded_file)
#             image_np = np.array(image)
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("Original Image")
#                 st.image(image, use_column_width=True)
            
#             with col2:
#                 st.subheader("Processed Result")
#                 with st.spinner("Detecting people..."):
#                     result_img, count = count_people(
#                         image_np, 
#                         confidence_threshold, 
#                         min_person_height, 
#                         nms_threshold
#                     )
#                     st.image(result_img, use_column_width=True)
#                     st.success(f"Detected {count} people in the image")
        
#         elif file_ext in ["mp4", "mov"]:
#             st.subheader("Video Processing")
            
#             tfile = tempfile.NamedTemporaryFile(delete=False)
#             tfile.write(uploaded_file.read())
            
#             video_col, result_col = st.columns(2)
            
#             with video_col:
#                 st.subheader("Original Video")
#                 video_bytes = uploaded_file.read()
#                 st.video(video_bytes)
            
#             with result_col:
#                 st.subheader("Processed Video")
#                 st.warning("Video processing may take some time...")
                
#                 temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
#                 temp_output.close()
                
#                 cap = cv2.VideoCapture(tfile.name)
#                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 fps = int(cap.get(cv2.CAP_PROP_FPS))
                
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 out = cv2.VideoWriter(
#                     temp_output.name, 
#                     fourcc, 
#                     fps, 
#                     (frame_width, frame_height)
#                 )
                
#                 progress_bar = st.progress(0)
#                 frame_count = 0
#                 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
                    
#                     processed_frame, _ = count_people(
#                         frame, 
#                         confidence_threshold, 
#                         min_person_height, 
#                         nms_threshold
#                     )
#                     out.write(processed_frame)
                    
#                     frame_count += 1
#                     progress = int((frame_count / total_frames) * 100)
#                     progress_bar.progress(min(progress, 100))
                
#                 cap.release()
#                 out.release()
                
#                 with open(temp_output.name, "rb") as f:
#                     st.video(f.read())
                
#                 os.unlink(tfile.name)
#                 os.unlink(temp_output.name)
                
#                 st.success("Video processing complete!")

# # Main app logic
# if input_method == "Use Camera":
#     camera_capture()
# else:
#     file_processing()

# # Add some app info
# st.markdown("---")
# st.markdown("""
#     ### How to Use:
#     1. Choose input method (upload file or use camera)
#     2. Adjust the detection parameters in the sidebar
#     3. View the processed results with people counted
    
#     ### Tips for Better Accuracy:
#     - Increase confidence threshold if you see false positives
#     - Decrease minimum height if people are being missed
#     - Adjust overlap threshold if boxes aren't merging properly
# """)






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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Preset configurations
PRESETS = {
    "High Precision (Crowded)": {
        "confidence": 0.8,
        "min_height": 20,
        "nms": 0.4,
        "help": "Best for dense crowds with potential false positives"
    },
    "Balanced (Default)": {
        "confidence": 0.7,
        "min_height": 15,
        "nms": 0.3,
        "help": "Good for most general scenarios"
    },
    "High Recall (Sparse)": {
        "confidence": 0.5,
        "min_height": 10,
        "nms": 0.2,
        "help": "Best for small or distant people"
    }
}

# Modified model loading with proper CUDA support check
@st.cache_resource
def load_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Check for CUDA support properly
    gpu_status = "⚠️ Using CPU"
    try:
        # First check if CUDA is available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            gpu_status = "⚡ GPU Acceleration Enabled"
            # Verify the backend was actually set
            if net.getPreferableBackend() != cv2.dnn.DNN_BACKEND_CUDA:
                gpu_status = "⚠️ CUDA not available - Using CPU"
        else:
            gpu_status = "⚠️ No CUDA devices found - Using CPU"
    except:
        gpu_status = "⚠️ CUDA not supported in this OpenCV build - Using CPU"
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers, gpu_status
net, classes, output_layers, gpu_status = load_model()

# Sidebar controls
with st.sidebar:
    st.title("Settings")
    st.caption(gpu_status)
    
    # Quick settings presets
    preset = st.selectbox(
        "Quick Presets",
        options=list(PRESETS.keys()),
        index=1,
        help="Select a preset configuration for different scenarios"
    )
    
    # Show preset description
    st.caption(PRESETS[preset]["help"])
    
    # Apply selected preset
    current_preset = PRESETS[preset]
    
    # Main controls
    input_method = st.radio(
        "Input Method",
        ["Upload File", "Use Camera"],
        index=0,
        help="Choose between uploading a file or using your webcam"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=current_preset["confidence"], 
        step=0.05,
        help="Higher values reduce false positives but may miss some people"
    )
    min_person_height = st.slider(
        "Minimum Person Height (%)", 
        min_value=5, 
        max_value=30, 
        value=current_preset["min_height"], 
        step=1,
        help="Filters out small detections that are likely not people"
    )
    nms_threshold = st.slider(
        "Overlap Threshold (NMS)", 
        min_value=0.1, 
        max_value=0.5, 
        value=current_preset["nms"], 
        step=0.05,
        help="Controls how much overlapping boxes are suppressed"
    )

# Main title and description
st.title("👥 Advanced People Counter")
st.markdown("""
    <style>
    .big-font {
        font-size:16px !important;
    }
    </style>
    <p class="big-font">Upload an image/video or use your camera to count people with advanced detection settings.</p>
""", unsafe_allow_html=True)

# People counting function with enhanced visualization
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
    
    # Draw results with enhanced visualization
    result_image = image.copy()
    people_count = 0
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw confidence percentage
            cv2.putText(
                result_image, 
                f"{confidences[i]:.1%}", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 255), 
                2
            )
            people_count += 1
    
    # Add count with background for better visibility
    cv2.rectangle(result_image, (10, 10), (250, 60), (0, 0, 0), -1)
    cv2.putText(
        result_image, 
        f"People Count: {people_count}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    return result_image, people_count

# Helper function for download buttons
def get_download_button(data, filename, file_type):
    if file_type == "image":
        return st.download_button(
            label="📥 Download Image",
            data=data,
            file_name=filename,
            mime="image/jpeg"
        )
    else:  # video
        return st.download_button(
            label="📥 Download Video",
            data=data,
            file_name=filename,
            mime="video/mp4"
        )

# Camera capture function with enhanced UI
def camera_capture():
    st.subheader("Live Camera Feed")
    with st.expander("Camera Settings", expanded=False):
        cam_resolution = st.selectbox("Resolution", ["640x480", "1280x720"], index=0)
        width, height = map(int, cam_resolution.split("x"))
    
    run_camera = st.checkbox("Start Camera", value=False, key="camera_toggle")
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(1)
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Placeholder for stats
    stats_placeholder = st.empty()
    fps_counter = 0
    start_time = time.time()
    people_counts = []
    
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
        
        # Calculate FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        fps = fps_counter / elapsed_time if elapsed_time > 0 else 0
        
        # Update stats
        people_counts.append(count)
        avg_count = np.mean(people_counts[-10:]) if people_counts else 0
        
        # Display frame
        FRAME_WINDOW.image(processed_frame, channels="RGB")
        
        # Update stats display
        stats_placeholder.markdown(f"""
            **Live Stats:**
            - Current Count: {count}
            - Average (last 10): {avg_count:.1f}
            - FPS: {fps:.1f}
        """)
        
        # Add small delay to prevent high CPU usage
        time.sleep(max(0.05, 1/30 - (time.time() - start_time - elapsed_time)))
    
    camera.release()
    if not run_camera:
        st.success("Camera session ended")

# File processing function with enhanced features
def file_processing():
    
    uploaded_file = st.file_uploader(
        "Choose an image or video", 
        type=["jpg", "jpeg", "png", "mp4", "mov"],
        help="Select an image or video file to analyze"
    )

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        if file_ext in ["jpg", "jpeg", "png"]:
            with st.spinner("Processing image..."):
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                tab1, tab2 = st.tabs(["Original", "Processed"])
                
                with tab1:
                    st.image(image, use_column_width=True, caption="Original Image")
                
                with tab2:
                    result_img, count = count_people(
                        image_np, 
                        confidence_threshold, 
                        min_person_height, 
                        nms_threshold
                    )
                    result_pil = Image.fromarray(result_img)
                    st.image(result_pil, use_column_width=True, caption=f"Detected {count} people")
                    
                    # Download button
                    buffered = BytesIO()
                    result_pil.save(buffered, format="JPEG", quality=90)
                    get_download_button(buffered.getvalue(), f"processed_{uploaded_file.name}", "image")
        
        elif file_ext in ["mp4", "mov"]:
            st.info("Video processing may take some time depending on length and settings")
            
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.close()
            
            # Process video
            temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_output.close()
            
            cap = cv2.VideoCapture(tfile.name)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                temp_output.name, 
                fourcc, 
                fps, 
                (frame_width, frame_height)
            )
            
            # Progress bar and stats
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_count = 0
            people_counts = []
            
            # Preview placeholder
            preview_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, count = count_people(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    confidence_threshold,
                    min_person_height,
                    nms_threshold
                )
                out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                
                # Update stats
                frame_count += 1
                people_counts.append(count)
                progress = int((frame_count / total_frames) * 100)
                progress_bar.progress(min(progress, 100))
                
                # Show preview every 50 frames
                if frame_count % 50 == 1:
                    preview_placeholder.image(processed_frame, caption=f"Frame {frame_count}/{total_frames}")
                
                # Update status
                status_text.markdown(f"""
                    **Processing Status:**
                    - Frames Processed: {frame_count}/{total_frames}
                    - Current Count: {count}
                    - Average Count: {np.mean(people_counts):.1f}
                    - Estimated Time Remaining: {(total_frames-frame_count)/(fps if fps > 0 else 30):.1f}s
                """)
            
            cap.release()
            out.release()
            
            # Show results
            st.success("Video processing complete!")
            
            # Display side-by-side comparison
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Video")
                st.video(uploaded_file)
            
            with col2:
                st.subheader("Processed Video")
                with open(temp_output.name, "rb") as f:
                    video_data = f.read()
                    st.video(video_data)
                    get_download_button(video_data, f"processed_{uploaded_file.name}", "video")
            
            # Clean up temp files
            os.unlink(tfile.name)
            os.unlink(temp_output.name)

# Main app logic
if input_method == "Use Camera":
    camera_capture()
else:
    file_processing()

# App info and documentation
with st.expander("ℹ️ About this App & Usage Tips", expanded=False):
    st.markdown("""
        ## People Counter Pro
        
        This application uses YOLOv3 object detection to count people in images, videos, or live camera feeds.
        
        ### Key Features:
        - **GPU Acceleration**: Automatically enabled if available
        - **Multiple Presets**: Optimized settings for different scenarios
        - **Detailed Statistics**: Real-time counts and performance metrics
        - **High-Quality Output**: Download processed files with detection results
        
        ### Advanced Tips:
        1. For crowded scenes, use **High Precision** preset
        2. For distant people, try **High Recall** with lower min height
        3. Monitor FPS in camera mode - lower resolution if needed
        4. Download results for further analysis
        
        ### Technical Notes:
        - Requires YOLOv3 weights and config files
        - Video processing time depends on length and hardware
        - All processing happens locally - no data is uploaded
    """)

# Add footer
st.markdown("---")
st.caption("""
    People Counter Pro v2.0 | 
    [GitHub Repository](https://github.com/example/people-counter) | 
    Created with Streamlit
""")