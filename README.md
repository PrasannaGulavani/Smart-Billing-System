# Smart-Billing-System

An intelligent automated billing system that uses computer vision (object detection with YOLOv5 and Faster R-CNN) combined with weight sensing to automatically identify products and calculate bills in a self-checkout environment.

## 📋 Project Overview

This system consists of three main components:
- **Python Backend**: Flask-based REST API with real-time computer vision processing
- **Weight Measurement System**: Raspberry Pi with HX711 load cell sensor for accurate product weight detection
- **Checkout UI**: Node.js + MongoDB backend with HTML/CSS/JavaScript frontend for billing management

## 🛠️ Technology Stack

### Backend (Python)
- **Flask**: Web framework for API endpoints
- **PyTorch**: Machine learning framework for object detection
  - Faster R-CNN with MobileNetV3 backbone
  - YOLOv5s custom model (`yolov5s.pt`)
- **OpenCV**: Computer vision library for image processing
- **Edge Impulse**: ML model deployment (`modelfile.eim`)
- **RPi.GPIO**: Raspberry Pi GPIO control
- **HX711 Library**: Weight sensor communication

### Frontend/Server (Node.js)
- **Express.js**: Web server framework
- **MongoDB**: NoSQL database for product and order storage
- **Mongoose**: MongoDB object modeling
- **CORS**: Cross-origin resource sharing
- **Body Parser**: Request body parsing

### Hardware
- Raspberry Pi with Python 3
- HX711 Load Cell ADC module (GPIO pins 5, 6)
- Camera module for object detection
- CUDA-capable GPU (optional, for faster inference)

## 📁 Project Structure

```
Smart-Billing-System/
├── app.py                    # Flask application with Faster R-CNN object detection
├── billing.py                # Billing logic with Edge Impulse integration
├── calibration.py            # Weight sensor calibration
├── Yolo.py                   # YOLOv5 object detection module
├── hx711.py                  # HX711 weight sensor driver
├── example_python3.py        # Example usage script
├── yolov5s.pt               # Pre-trained YOLOv5s model weights
├── modelfile.eim            # Edge Impulse model file
├── Rasp config.txt          # Raspberry Pi configuration notes
├── README.md                # This file
└── CheckoutUI/
    ├── client/              # Frontend (HTML/CSS/JavaScript)
    │   ├── index.html       # Main checkout page
    │   ├── add.html         # Product addition page
    │   └── asset/
    │       ├── css/
    │       │   └── style.css
    │       ├── img/         # UI images
    │       └── scripts/
    │           └── script.js
    └── server/              # Node.js backend
        ├── server.js        # Express server with API routes
        └── package.json     # Node.js dependencies
```

## 🚀 Getting Started

### Prerequisites
- Raspberry Pi (4B or higher recommended)
- Python 3.7+
- Node.js 16.x
- MongoDB (local or cloud instance)
- Camera module for Raspberry Pi
- HX711 Load Cell module
- CUDA toolkit (optional, for GPU acceleration)

### Python Setup

1. **Install Python dependencies**:
   ```bash
   pip install flask torch torchvision opencv-python pillow numpy
   pip install RPi.GPIO
   pip install requests
   pip install edge-impulse-linux
   ```

2. **Calibrate the weight sensor**:
   ```bash
   python calibration.py
   ```
   This will configure the HX711 sensor and set the weight calibration ratio.

3. **Configure Raspberry Pi GPIO**:
   - HX711 DATA pin → GPIO 5
   - HX711 CLOCK pin → GPIO 6
   - Update pin configuration in `app.py` and `billing.py` as needed

### Node.js Server Setup

1. **Install dependencies**:
   ```bash
   cd CheckoutUI/server
   npm install
   ```

2. **Configure MongoDB**:
   - Update connection string in `server.js`
   - Ensure MongoDB is running locally or on your cloud provider

3. **Start the server**:
   ```bash
   npm start
   ```
   Server will run on port 3000 (configurable via PORT environment variable)

## 🎯 Main Features

### Object Detection
- **Faster R-CNN**: Fast object detection with MobileNetV3 backbone
- **YOLOv5s**: Custom-trained YOLOv5 small model for specific product detection
- Real-time video stream processing via Flask API

### Weight Measurement
- **HX711 ADC**: Precise weight measurement from load cells
- **Calibration**: Dynamic calibration support in `calibration.py`
- **Integration**: Weight data used for product verification and billing

### Billing System
- Automatic product identification via computer vision
- Weight-based verification
- Real-time order processing
- MongoDB-based order history
- REST API for checkout operations

### Checkout UI
- Clean HTML interface for checkout process
- Real-time product addition
- Order summary and billing display
- Responsive design with CSS styling

## 📡 API Endpoints

### Flask Backend (Python)
- `GET /`: Main camera feed endpoint
- `GET /video_feed`: Streaming video with object detection overlay

### Node.js Server
- `GET /`: Home endpoint
- Product management and order processing endpoints (defined in `server.js`)

## ⚙️ Configuration

### Weight Sensor Calibration
Edit `calibration.py` to adjust:
- GPIO pins for HX711
- Reference weight values
- Calibration ratio

### Model Selection
- Switch between YOLOv5s and Faster R-CNN in `app.py`
- Load custom weights in `Yolo.py` via the `model_path` variable

### Device Configuration
- Enable CUDA in `app.py`: Device is auto-detected
- Set Flask debug mode and port in `app.py`

## 🐛 Troubleshooting

**Weight sensor not reading correctly**:
- Run `calibration.py` to recalibrate
- Check GPIO pin connections
- Verify HX711 module is properly connected

**Object detection is slow**:
- Ensure CUDA is properly installed (for GPU acceleration)
- Reduce frame resolution
- Switch to lighter model (YOLOv5s is recommended)

**MongoDB connection failed**:
- Verify MongoDB service is running
- Check connection string in `server.js`
- Ensure network connectivity for cloud MongoDB

## 📝 Usage Example

1. **Start the Flask backend**:
   ```bash
   python app.py
   ```

2. **In another terminal, start the Node.js server**:
   ```bash
   cd CheckoutUI/server
   npm start
   ```

3. **Access the checkout UI**:
   - Open `http://localhost:3000` in a web browser
   - Navigate to `index.html` for main checkout
   - Use `add.html` for product management

4. **Monitor console output**:
   - Flask will display detected objects
   - Check weight measurements from HX711
   - Monitor billing transactions in server logs

## 📄 License

ISC

## 👤 Author

Shebin Jose Jacob
