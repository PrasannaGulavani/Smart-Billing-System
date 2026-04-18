from flask import Flask, render_template, Response
import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
from os import environ

import threading
import random
import time

app = Flask(__name__)

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the pre-trained model
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
model.eval()
model.to(device)

detected_objects=None
unique_detected_objects = dict()
newobjwt=0
################################

import RPi.GPIO as GPIO
import time
import sys
from hx711 import HX711

if sys.version_info[0] != 3:
    raise Exception("Python 3 is required.")


hx = HX711(5, 6)

def cleanAndExit():
    print("Cleaning...")
    GPIO.cleanup()
    print("Bye!")
    sys.exit()


def setup():
    """
    code run once
    """
    hx.set_offset(8069595.125)
    hx.set_scale(448.52)
    
###############################
def update_data():
    global unique_detected_objects,newobjwt
    while True:
        '''
        newobjwt= random.randint(50, 200)
        #print(newobjwt)
        time.sleep(0.5)  # Update every 5 seconds
        '''
        try:
            val = hx.get_grams()
            newobjwt=abs(int(val))
            print(newobjwt)

            hx.power_down()
            time.sleep(.001)
            hx.power_up()

            time.sleep(2)
        except (KeyboardInterrupt, SystemExit):
            cleanAndExit()

setup()

# COCO dataset class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Initialize the video capture object
camera = cv2.VideoCapture(0)  # Use 0 for the default webcam
width, height = 420, 300
print("Starting video capture...")
if not camera.isOpened():
    print("Error: Could not open video device")

# Function to preprocess the frame
def preprocess(frame):
    frame = cv2.resize(frame, (480, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_tensor = F.to_tensor(pil_image).unsqueeze(0).to(device)
    return image_tensor

# Function to draw bounding boxes and list detected objects
def draw_boxes_and_list_objects(frame, boxes, labels, scores, threshold=0.7):
    global unique_detected_objects,newobjwt
    detected_objects = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            text = f'{COCO_INSTANCE_CATEGORY_NAMES[label.item()]}: {score:.2f}'
            if COCO_INSTANCE_CATEGORY_NAMES[label.item()] in ['banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake']:
                unique_detected_objects[COCO_INSTANCE_CATEGORY_NAMES[label.item()]]=newobjwt
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detected_objects.append(COCO_INSTANCE_CATEGORY_NAMES[label.item()])
    return detected_objects

# Generate frames for the video feed
def generate_frames():
    global detected_objects
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            image_tensor = preprocess(frame)

            with torch.no_grad():
                predictions = model(image_tensor)

            boxes = predictions[0]['boxes'].cpu()
            labels = predictions[0]['labels'].cpu()
            scores = predictions[0]['scores'].cpu()
                
            detected_objects = draw_boxes_and_list_objects(frame, boxes, labels, scores)
            #print(detected_objects)
            #newobjwt=detected_objects
            #Update unique detected objects list
            #unique_detected_objects.update(detected_objects)
            
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def stream_detected_objects1():
    while True:
        if detected_objects==None:
            break
        else:
            detected_objects = draw_boxes_and_list_objects(frame, boxes, labels, scores)

            yield f'data: {json.dumps(detected_objects)}\n\n'
            time.sleep(1)

def stream_detected_objects1():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            image_tensor = preprocess(frame)

            with torch.no_grad():
                predictions = model(image_tensor)

            boxes = predictions[0]['boxes'].cpu()
            labels = predictions[0]['labels'].cpu()
            scores = predictions[0]['scores'].cpu()

            detected_objects = draw_boxes_and_list_objects(frame, boxes, labels, scores)

            yield f'data: {json.dumps(detected_objects)}\n\n'
            time.sleep(1)
            
@app.route('/detected_objects')
def detected_objects():
    html_response = '<table class="table"><thead><tr><th scope="col">Product</th><th scope="col">Weight(Gm)</th><th scope="col">Price</th><th scope="col">Total</th></tr></thead><tbody>'
    Allprice=0
    #for obj in unique_detected_objects:
    for key, value in unique_detected_objects.items():
        wtg=value
        price=1
        if key=='banana':
             price=1
        if key=='apple':
             price=2
        if key=='sandwich':
             price=2
        if key=='orange':
             price=2
        if key=='broccoli':
             price=1
        if key=='carrot':
             price=1
        if key=='hot dog':
             price=1
        if key=='pizza':
             price=5
        if key=='donut':
             price=2
        if key=='donut':
             price=2             

        Tprice=(wtg*price)/10
        Allprice=Allprice+Tprice
        html_response += f'<tr><td>{key}</td><td>{wtg}Gm</td><td>{price}</td><td>{Tprice}</td></tr>'
    
    html_response += f'<tr><td> </td><td> </td><td>Total Price</td><td>{Allprice}</td></tr>'
    html_response += '</tbody></table>'
    return html_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/Data_clean')
def Data_clean():
    global unique_detected_objects,newobjwt
    unique_detected_objects.clear()
    newobjwt=0
    html_response = '<table class="table"><thead><tr><th scope="col">Product</th><th scope="col">Weight</th><th scope="col">Price</th><th scope="col">Total</th></tr></thead><tbody>'
    Allprice=0    
    html_response += f'<tr><td> </td><td> </td><td>Total Price</td><td>{Allprice}</td></tr>'
    html_response += '</tbody></table>'
    return html_response

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


'''
@app.route('/detected_objects')
def detected_objects():
    return Response(stream_with_context(stream_detected_objects()), content_type='text/event-stream')
'''

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    update_thread = threading.Thread(target=update_data)
    update_thread.daemon = True
    update_thread.start()
    
    HOST = environ.get('SERVER_HOST', '0.0.0.0')
    try:
       PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
       PORT = 5555
    app.run(HOST, PORT)
