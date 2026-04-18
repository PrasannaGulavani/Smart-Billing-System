import torch
import cv2
import numpy as np

# Load pre-trained YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load custom YOLOv5 model
model_path = 'yolov5s.pt'  # Update this with your model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Print the class names
class_names = model.names
print(class_names)
print("List of objects that YOLOv5s can detect:")
for idx, class_name in enumerate(class_names):
    print(f"{idx}: {class_name}")
    
# Set the model to evaluation mode
model.eval()

# Define the vehicle classes (COCO dataset class IDs for vehicles)
vehicle_classes = [2, 3, 5, 7]  #
vehicle_classes_dir={2:'car',3:'motorcycle',5:'bus',7:'truck'}

# Function to draw bounding boxes
def draw_boxes(img, boxes, labels, confidences):
    for box, label, confidence in zip(boxes, labels, confidences):
        if label in vehicle_classes:
            color = (0, 255, 0)  # Green for vehicles
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img, f'{label} {confidence:.2f} {vehicle_classes_dir[label]}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv5 model on the frame
        results = model(frame)

        # Extract results
        boxes = results.xyxy[0][:, :4].numpy()
        confidences = results.xyxy[0][:, 4].numpy()
        labels = results.xyxy[0][:, 5].numpy()

        # Draw bounding boxes
        draw_boxes(frame, boxes, labels, confidences)

        # Display the frame with detection boxes
        cv2.imshow('Vehicle Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
