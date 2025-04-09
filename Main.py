# import cv2
# import numpy as np

# # Load YOLOv3 model
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Get output layer names
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Initialize camera
# cap = cv2.VideoCapture(2)  # 0 for default camera

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     height, width, channels = frame.shape

#     # Detect objects (YOLO expects blob input)
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Process detections
#     class_ids = []
#     confidences = []
#     boxes = []
#     people_count = 0

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             # Filter for "person" class (class_id = 0 in COCO dataset)
#             if class_id == 0 and confidence > 0.5:  # Confidence threshold
#                 people_count += 1
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Non-max suppression to avoid duplicate boxes
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Draw bounding boxes and count
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display people count
#     cv2.putText(
#         frame,
#         f"People: {people_count}",
#         (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 0, 255),
#         2,
#     )

#     # Show output
#     cv2.imshow("Crowd Counting", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Hello")


# import cv2
# import numpy as np

# def count_people_in_image(image_path, confidence_threshold=0.6, nms_threshold=0.4):
#     # Load YOLOv3 model
#     net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#     with open("coco.names", "r") as f:
#         classes = [line.strip() for line in f.readlines()]
    
#     # Get output layers
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#     # Load image
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image from {image_path}")
#         return
    
#     height, width = img.shape[:2]

#     # Preprocess
#     blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Detection parameters
#     people_count = 0
#     boxes = []
#     confidences = []
#     class_ids = []

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             # Only detect people with high confidence
#             if class_id == 0 and confidence > confidence_threshold:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Filter small detections (likely false positives)
#                 if w * h > (width * height) / 1000:  # At least 0.1% of image area
#                     boxes.append([center_x, center_y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#     # Apply Non-Max Suppression aggressively
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

#     # Final count and drawing
#     people_count = len(indices)
    
#     # Draw results
#     if len(indices) > 0:
#         for i in indices.flatten():
#             x, y, w, h = boxes[i]
#             cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)

#     # Display count
#     cv2.putText(img, f"Accurate Count: {people_count}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     # Save and show
#     output_path = "accurate_result.jpg"
#     cv2.imwrite(output_path, img)
    
#     cv2.imshow("Accurate People Counting", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     print(f"Final Count: {people_count} people")
#     print(f"Saved result to {output_path}")

# # Usage with strict parameters
# count_people_in_image("garden.jpg", confidence_threshold=0.7, nms_threshold=0.3)



import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize camera (try different indices if needed)
cap = cv2.VideoCapture(1)  # 0, 1, or 2 for different cameras

# Optimized parameters
CONFIDENCE_THRESHOLD = 0.7  # Increased from 0.5 to reduce false positives
NMS_THRESHOLD = 0.3         # More aggressive suppression of overlapping boxes
MIN_PERSON_HEIGHT = 0.15    # Minimum height (relative to frame height) to consider as valid detection

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    height, width = frame.shape[:2]
    min_h = int(height * MIN_PERSON_HEIGHT)  # Calculate absolute minimum height

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    people_count = 0
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only detect people with high confidence
            if class_id == 0 and confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Filter small detections and invalid boxes
                if h >= min_h and w > 10 and h > 10:
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Ensure boxes stay within frame boundaries
                    x, y = max(0, x), max(0, y)
                    w, h = min(width - x, w), min(height - y, h)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Final count of valid people
    people_count = len(indices)
    
    # Draw bounding boxes and count
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display confidence percentage on each detection
            cv2.putText(frame, f"{confidences[i]:.1%}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display people count with background for better visibility
    cv2.rectangle(frame, (10, 10), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"People: {people_count} (Conf: {CONFIDENCE_THRESHOLD:.0%})",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Accurate People Counting", frame)

    # Key controls for adjusting parameters in real-time
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') and CONFIDENCE_THRESHOLD < 0.9:
        CONFIDENCE_THRESHOLD += 0.05
    elif key == ord('-') and CONFIDENCE_THRESHOLD > 0.3:
        CONFIDENCE_THRESHOLD -= 0.05
    elif key == ord('h') and MIN_PERSON_HEIGHT < 0.3:
        MIN_PERSON_HEIGHT += 0.02
    elif key == ord('l') and MIN_PERSON_HEIGHT > 0.05:
        MIN_PERSON_HEIGHT -= 0.02

cap.release()
cv2.destroyAllWindows()