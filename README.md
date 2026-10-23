# Safety Vest Detection System

This system uses YOLOv8 to detect safety vests in images and videos, with a Flask web interface.

## Features

- Real-time safety vest detection using YOLOv8
- Image upload for detection on existing photos
- Live webcam feed for real-time detection
- Notification system for vest detection alerts
- Visualization with bounding boxes around detected vests

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)
- Webcam for real-time detection

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask server:
   ```
   python test.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. On the home page, you'll have two options:
   - **Upload Image**: Click to upload an existing image for vest detection
   - **Live Camera**: Start webcam detection in real-time

4. For image upload:
   - The system will display the original image and processed image with bounding boxes
   - Detection results will be shown below the images

5. For live camera:
   - Allow camera access when prompted
   - The system will show real-time detection with bounding boxes
   - Notifications will appear when vests are detected

## Model Information

The system uses a YOLOv8 model trained on a custom dataset of safety vests. The model can detect two classes:
- `vest`: Workers wearing safety vests
- `no-vest`: Workers not wearing safety vests

## Project Structure

- `test.py`: Flask server with YOLO inference
- `front_end/`: Web interface files
  - `index.html`: Main interface with image upload and webcam detection
  - `script.js`: JavaScript for image upload, camera access and API calls
  - `styles.css`: Styling for the web pages
- `train/`, `valid/`, `test/`: Datasets used for training
- `*.pt` files: Trained YOLO models

## License

MIT "# vest-" 
