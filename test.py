import cv2
import torch
from ultralytics import YOLO
from flask import Flask, request, jsonify
import numpy as np
import base64
import os
import io
import time
import imageio
from flask_cors import CORS # type: ignore

# Set up Flask app with simple configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained YOLOv8 model
try:
    model = YOLO("test1.pt")
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Using CUDA for inference")
    else:
        print("Using CPU for inference")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for detections

def detect_objects(frame):
    try:
        # Run inference (assumes frame is in BGR format)
        results = model(frame)[0]
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs

            # Filter detections based on confidence threshold
            valid_indices = confidences > CONFIDENCE_THRESHOLD
            filtered_boxes = boxes[valid_indices]
            filtered_confidences = confidences[valid_indices]
            filtered_class_ids = class_ids[valid_indices]

            for i in range(len(filtered_boxes)):
                detections.append({
                    'class': model.names[filtered_class_ids[i]],
                    'confidence': float(filtered_confidences[i]),
                    'box': filtered_boxes[i].tolist()
                })
        # Annotate the frame using YOLO's built-in plot()
        annotated_frame = results.plot()
        return annotated_frame, detections
    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")
        return frame, []

@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Safety Vest Detection System</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #1E40AF; /* Deep blue for professionalism */
      --secondary: #3B82F6; /* Lighter blue for highlights */
      --accent: #60A5FA; /* Soft blue for interactive elements */
      --success: #10B981; /* Green for positive states */
      --warning: #EF4444; /* Red for errors */
      --background: #F9FAFB; /* Light gray background */
      --surface: #FFFFFF; /* White for cards */
      --text-primary: #111827; /* Dark gray for text */
      --text-secondary: #6B7280; /* Muted gray for secondary text */
      --border: #E5E7EB; /* Light border color */
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--background);
      color: var(--text-primary);
      line-height: 1.5;
      font-size: 16px;
    }

    .container {
      max-width: 1280px;
      margin: 0 auto;
      padding: 2rem 1rem;
    }

    header {
      text-align: center;
      margin-bottom: 3rem;
      padding: 2rem;
      background: var(--surface);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }

    h1 {
      font-size: 2.25rem;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 0.5rem;
    }

    .subtitle {
      font-size: 1rem;
      color: var(--text-secondary);
      font-weight: 400;
    }

    .card {
      background: var(--surface);
      border-radius: 12px;
      box-shadow: var(--shadow);
      padding: 2rem;
      margin-bottom: 2rem;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .card-title {
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .card-title svg {
      width: 24px;
      height: 24px;
      color: var(--primary);
    }

    .options-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.5rem;
      margin-bottom: 2rem;
    }

    .option-card {
      background: var(--surface);
      border-radius: 12px;
      padding: 2rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s ease;
      border: 1px solid var(--border);
    }

    .option-card:hover {
      transform: translateY(-5px);
      border-color: var(--secondary);
      box-shadow: var(--shadow);
    }

    .option-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
      color: var(--primary);
      transition: color 0.3s ease;
    }

    .option-card:hover .option-icon {
      color: var(--secondary);
    }

    .option-title {
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 0.5rem;
    }

    .option-desc {
      font-size: 0.875rem;
      color: var(--text-secondary);
    }

    .result-container {
      display: none;
    }

    .media-container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }

    @media (max-width: 768px) {
      .media-container {
        grid-template-columns: 1fr;
      }
    }

    .media-box {
      border-radius: 8px;
      overflow: hidden;
      position: relative;
      background: #F3F4F6;
      min-height: 300px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--border);
    }

    .media-box img, .media-box video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .media-label {
      position: absolute;
      top: 0;
      left: 0;
      background: rgba(17, 24, 39, 0.8);
      color: white;
      padding: 0.5rem 1rem;
      border-bottom-right-radius: 8px;
      font-size: 0.875rem;
      font-weight: 500;
    }

    .detection-info {
      background: #F9FAFB;
      border-radius: 8px;
      padding: 1.5rem;
      border: 1px solid var(--border);
    }

    .detection-stats {
      display: flex;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .stat-box {
      flex: 1;
      background: var(--surface);
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
      border: 1px solid var(--border);
      transition: transform 0.2s ease;
    }

    .stat-box:hover {
      transform: translateY(-2px);
    }

    .stat-value {
      font-size: 1.75rem;
      font-weight: 600;
      color: var(--primary);
    }

    .stat-label {
      font-size: 0.875rem;
      color: var(--text-secondary);
      margin-top: 0.25rem;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      font-weight: 500;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
      gap: 0.5rem;
      text-decoration: none;
    }

    .btn-primary {
      background: var(--primary);
      color: white;
    }

    .btn-primary:hover {
      background: var(--secondary);
    }

    .btn-outline {
      background: transparent;
      color: var(--primary);
      border: 1px solid var(--primary);
    }

    .btn-outline:hover {
      background: rgba(30, 64, 175, 0.05);
      border-color: var(--secondary);
      color: var(--secondary);
    }

    .status-badge {
      position: absolute;
      top: 10px;
      right: 10px;
      background: var(--warning);
      color: white;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
      z-index: 10;
    }

    .status-active {
      background: var(--success);
    }

    .hidden {
      display: none;
    }

    .loading {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 1rem;
      padding: 2rem;
    }

    .spinner {
      width: 32px;
      height: 32px;
      border: 3px solid rgba(30, 64, 175, 0.2);
      border-radius: 50%;
      border-top-color: var(--primary);
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .detection-list {
      margin-top: 1rem;
    }

    .detection-item {
      display: flex;
      align-items: center;
      padding: 0.75rem 0;
      border-bottom: 1px solid var(--border);
    }

    .detection-item:last-child {
      border-bottom: none;
    }

    .detection-class {
      width: 120px;
      font-weight: 500;
      color: var(--text-primary);
    }

    .detection-confidence {
      flex: 1;
      padding: 0 1rem;
    }

    .confidence-bar {
      height: 6px;
      background: #E5E7EB;
      border-radius: 9999px;
      overflow: hidden;
    }

    .confidence-fill {
      height: 100%;
      background: var(--secondary);
      border-radius: 9999px;
      transition: width 0.3s ease;
    }

    .detection-percent {
      width: 60px;
      text-align: right;
      font-size: 0.875rem;
      color: var(--text-secondary);
    }

    .vest-detected {
      color: var(--success);
    }

    .no-vest-detected {
      color: var(--warning);
    }

    /* Improved responsiveness */
    @media (max-width: 1024px) {
      .container {
        padding: 1.5rem;
      }

      h1 {
        font-size: 2rem;
      }

      .card-title {
        font-size: 1.25rem;
      }
    }

    @media (max-width: 640px) {
      .options-grid {
        grid-template-columns: 1fr;
      }

      .stat-box {
        padding: 0.75rem;
      }

      .stat-value {
        font-size: 1.5rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Safety Vest Detection System</h1>
      <p class="subtitle">Advanced AI-powered detection for workplace safety</p>
    </header>

    <div class="card">
      <h2 class="card-title">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        Select Detection Method
      </h2>
      <div class="options-grid">
        <div class="option-card" id="uploadImageButton">
          <div class="option-icon">📷</div>
          <h3 class="option-title">Image Detection</h3>
          <p class="option-desc">Upload an image to detect safety vests instantly</p>
          <input type="file" id="imageUpload" accept="image/*" class="hidden">
        </div>

        <div class="option-card" id="uploadVideoButton">
          <div class="option-icon">🎞️</div>
          <h3 class="option-title">Video Detection</h3>
          <p class="option-desc">Process a video file for safety vest detection</p>
          <input type="file" id="videoUpload" accept="video/*" class="hidden">
        </div>

        <div class="option-card" id="cameraButton">
          <div class="option-icon">🎥</div>
          <h3 class="option-title">Live Detection</h3>
          <p class="option-desc">Real-time detection using your device's camera</p>
        </div>
      </div>
    </div>

    <!-- Image Detection Result -->
    <div class="card result-container" id="uploadResult">
      <h2 class="card-title">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        Image Detection Results
      </h2>

      <div class="media-container">
        <div class="media-box">
          <span class="media-label">Original Image</span>
          <img id="originalImage" alt="Original Image">
        </div>
        <div class="media-box">
          <span class="media-label">Processed Image</span>
          <img id="processedImage" alt="Processed Image">
        </div>
      </div>

      <div class="detection-info">
        <div class="loading" id="imageLoading">
          <div class="spinner"></div>
          <p>Processing image...</p>
        </div>
        <div id="imageResults">
          <div class="detection-stats">
            <div class="stat-box">
              <div class="stat-value" id="vestCount">0</div>
              <div class="stat-label">Safety Vests Detected</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" id="noVestCount">0</div>
              <div class="stat-label">No Vest Detected</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" id="confidenceLevel">0%</div>
              <div class="stat-label">Max Confidence</div>
            </div>
          </div>

          <h4 style="font-weight: 600; margin-bottom: 1rem;">Detection Details</h4>
          <div class="detection-list" id="detectionList"></div>
        </div>
      </div>
    </div>

    <!-- Video Detection Result -->
    <div class="card result-container" id="uploadVideoResult">
      <h2 class="card-title">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        Video Detection Results
      </h2>

      <div class="media-container">
        <div class="media-box">
          <span class="media-label">Original Video</span>
          <video id="originalVideo" controls></video>
        </div>
        <div class="media-box">
          <span class="media-label">Processed Video</span>
          <video id="processedVideo" controls></video>
        </div>
      </div>

      <div class="detection-info">
        <div class="loading" id="videoLoading">
          <div class="spinner"></div>
          <p>Processing video...</p>
        </div>
        <div id="videoResults">
          <a class="btn btn-outline" id="downloadVideoBtn" href="#">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
              <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
              <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
            </svg>
            Download Processed Video
          </a>
        </div>
      </div>
    </div>

    <!-- Live Camera Detection -->
    <div class="card result-container" id="cameraResult">
      <h2 class="card-title">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        Live Camera Detection
      </h2>

      <div class="media-box" id="video-feed">
        <span class="status-badge">Inactive</span>
        <p id="cameraPlaceholder">Camera feed will appear here</p>
      </div>

      <div class="detection-info">
        <div id="cameraResults">
          <div class="detection-stats">
            <div class="stat-box">
              <div class="stat-value" id="liveVestCount">0</div>
              <div class="stat-label">Safety Vests Detected</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" id="liveNoVestCount">0</div>
              <div class="stat-label">No Vest Detected</div>
            </div>
          </div>
        </div>

        <button class="btn btn-primary" id="stopCameraBtn">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M5.5 3.5A1.5 1.5 0 0 1 7 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5zm5 0A1.5 1.5 0 0 1 12 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5z"/>
          </svg>
          Stop Camera
        </button>
      </div>
    </div>
  </div>

  <script>
    // Your existing JavaScript code remains unchanged
    // DOM Elements
    const uploadImageBtn = document.getElementById('uploadImageButton');
    const uploadVideoBtn = document.getElementById('uploadVideoButton');
    const cameraBtn = document.getElementById('cameraButton');
    const imageUpload = document.getElementById('imageUpload');
    const videoUpload = document.getElementById('videoUpload');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');
    
    // Result containers
    const imageResult = document.getElementById('uploadResult');
    const videoResult = document.getElementById('uploadVideoResult');
    const cameraResult = document.getElementById('cameraResult');
    
    // Loading indicators
    const imageLoading = document.getElementById('imageLoading');
    const imageResults = document.getElementById('imageResults');
    const videoLoading = document.getElementById('videoLoading');
    const videoResults = document.getElementById('videoResults');
    
    // Stats elements
    const vestCountEl = document.getElementById('vestCount');
    const noVestCountEl = document.getElementById('noVestCount');
    const confidenceLevelEl = document.getElementById('confidenceLevel');
    const liveVestCountEl = document.getElementById('liveVestCount');
    const liveNoVestCountEl = document.getElementById('liveNoVestCount');
    const detectionListEl = document.getElementById('detectionList');
    
    // Video elements
    const videoFeed = document.getElementById('video-feed');
    const cameraPlaceholder = document.getElementById('cameraPlaceholder');
    
    // State variables
    let cameraStream = null;
    let processingInterval = null;
    
    // Event Listeners
    uploadImageBtn.addEventListener('click', () => imageUpload.click());
    uploadVideoBtn.addEventListener('click', () => videoUpload.click());
    
    imageUpload.addEventListener('change', handleImageUpload);
    videoUpload.addEventListener('change', handleVideoUpload);
    cameraBtn.addEventListener('click', startCameraDetection);
    stopCameraBtn.addEventListener('click', stopCameraDetection);
    downloadVideoBtn.addEventListener('click', downloadProcessedVideo);
    
    // Functions
    function handleImageUpload(event) {
      if (event.target.files.length === 0) return;
      
      const file = event.target.files[0];
      const reader = new FileReader();
      
      reader.onload = function(e) {
        // Show original image
        document.getElementById('originalImage').src = e.target.result;
        
        // Show loading state
        imageResult.style.display = 'block';
        videoResult.style.display = 'none';
        cameraResult.style.display = 'none';
        imageLoading.style.display = 'flex';
        imageResults.style.display = 'none';
        
        // Process image
        fetch('/predict_vest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: e.target.result })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            // Show processed image
            document.getElementById('processedImage').src = data.processed_image;
            
            // Update stats
            const vestCount = data.detections.filter(d => d.class === 'vest').length;
            const noVestCount = data.detections.filter(d => d.class === 'no-vest').length;
            const maxConfidence = data.detections.reduce((max, d) => d.confidence > max ? d.confidence : max, 0);
            
            vestCountEl.textContent = vestCount;
            noVestCountEl.textContent = noVestCount;
            confidenceLevelEl.textContent = `${Math.round(maxConfidence * 100)}%`;
            
            // Update detection list
            detectionListEl.innerHTML = '';
            data.detections.forEach(detection => {
              const detectionItem = document.createElement('div');
              detectionItem.className = 'detection-item';
              
              detectionItem.innerHTML = `
                <div class="detection-class ${detection.class === 'vest' ? 'vest-detected' : 'no-vest-detected'}">
                  ${detection.class === 'vest' ? 'Vest' : 'No Vest'}
                </div>
                <div class="detection-confidence">
                  <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${detection.confidence * 100}%"></div>
                  </div>
                </div>
                <div class="detection-percent">${Math.round(detection.confidence * 100)}%</div>
              `;
              
              detectionListEl.appendChild(detectionItem);
            });
            
            // Show results
            imageLoading.style.display = 'none';
            imageResults.style.display = 'block';
          } else {
            showError('imageResults', data.error || 'Failed to process image');
          }
        })
        .catch(err => {
          showError('imageResults', err.message);
        });
      };
      
      reader.readAsDataURL(file);
    }
    
    function handleVideoUpload(event) {
      if (event.target.files.length === 0) return;
      
      const file = event.target.files[0];
      const reader = new FileReader();
      
      reader.onload = function(e) {
        // Show original video
        const originalVideo = document.getElementById('originalVideo');
        originalVideo.src = e.target.result;
        
        // Show loading state
        imageResult.style.display = 'none';
        videoResult.style.display = 'block';
        cameraResult.style.display = 'none';
        videoLoading.style.display = 'flex';
        videoResults.style.display = 'none';
        
        // Process video
        fetch('/predict_video_file', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ video: e.target.result })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            // Show processed video
            document.getElementById('processedVideo').src = data.processed_video_url;
            
            // Enable download button
            downloadVideoBtn.onclick = () => {
              window.location.href = data.processed_video_url;
            };
            
            // Show results
            videoLoading.style.display = 'none';
            videoResults.style.display = 'block';
          } else {
            showError('videoResults', data.error || 'Failed to process video');
          }
        })
        .catch(err => {
          showError('videoResults', err.message);
        });
      };
      
      reader.readAsDataURL(file);
    }
    
    function startCameraDetection() {
    // Show camera container
    imageResult.style.display = 'none';
    videoResult.style.display = 'none';
    cameraResult.style.display = 'block';
    
    // Clear any existing content
    videoFeed.innerHTML = `
        <span class="status-badge">Inactive</span>
        <p id="cameraPlaceholder">Starting camera...</p>
    `;
    
    // Request camera access
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            cameraStream = stream;
            
            // Create video element
            const video = document.createElement('video');
            video.srcObject = stream;
            video.autoplay = true;
            video.playsInline = true;
            
            // Update UI
            videoFeed.innerHTML = '';
            const statusBadge = document.createElement('span');
            statusBadge.className = 'status-badge status-active';
            statusBadge.textContent = 'Active';
            videoFeed.appendChild(statusBadge);
            videoFeed.appendChild(video);
            
            // Start processing frames
            startFrameProcessing(video);
            
            // Handle when the video stream ends
            stream.getVideoTracks()[0].onended = () => {
                stopCameraDetection();
            };
        })
        .catch(err => {
            videoFeed.innerHTML = `<p style="color: var(--warning)">Error: ${err.message}</p>`;
        });
}

function startFrameProcessing(video) {
    // Clear any existing interval
    if (processingInterval) {
        clearInterval(processingInterval);
    }
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        processingInterval = setInterval(() => {
            if (!video || video.paused || video.ended) {
                clearInterval(processingInterval);
                return;
            }
            
            try {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg');
                
                fetch('/predict_vest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateDetectionStats(data.detections);
                        showProcessedFrame(data.processed_image);
                    }
                })
                .catch(console.error);
            } catch (e) {
                console.error("Frame processing error:", e);
                clearInterval(processingInterval);
            }
        }, 1000);
    };
}

function stopCameraDetection() {
    // Stop the camera stream
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => {
            track.stop();
        });
        cameraStream = null;
    }
    
    // Clear the processing interval
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    
    // Reset the UI
    videoFeed.innerHTML = `
        <span class="status-badge">Inactive</span>
        <p id="cameraPlaceholder">Camera feed will appear here</p>
    `;
    
    // Reset counters
    resetDetectionStats();
}

function updateDetectionStats(detections) {
    if (!liveVestCountEl || !liveNoVestCountEl) return;
    
    const vestCount = detections.filter(d => d.class === 'vest').length;
    const noVestCount = detections.filter(d => d.class === 'no-vest').length;
    
    liveVestCountEl.textContent = vestCount;
    liveNoVestCountEl.textContent = noVestCount;
}

function showProcessedFrame(imageSrc) {
    const existingOverlay = videoFeed.querySelector('img');
    if (existingOverlay) existingOverlay.remove();
    
    const overlay = document.createElement('img');
    overlay.src = imageSrc;
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.objectFit = 'cover';
    
    videoFeed.appendChild(overlay);
}

function resetDetectionStats() {
    if (liveVestCountEl && liveNoVestCountEl) {
        liveVestCountEl.textContent = '0';
        liveNoVestCountEl.textContent = '0';
    }
}
    function downloadProcessedVideo() {
      const processedVideo = document.getElementById('processedVideo');
      if (processedVideo.src) {
        window.location.href = processedVideo.src;
      }
    }
    
    function showError(containerId, message) {
      const container = document.getElementById(containerId);
      container.innerHTML = `
        <div style="color: var(--warning); padding: 1rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
          <strong>Error:</strong> ${message}
        </div>
      `;
      container.style.display = 'block';
    }
    
    // Initialize UI
    imageResults.style.display = 'none';
    videoResults.style.display = 'none';
  </script>
</body>
</html>
    """

@app.route('/predict_vest', methods=['POST'])
def predict_vest():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided', 'success': False}), 400
        try:
            encoded_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({'error': 'Failed to decode image', 'success': False}), 400
        except Exception as e:
            return jsonify({'error': f'Error decoding image: {str(e)}', 'success': False}), 400

        processed_img, detections = detect_objects(img)
        try:
            _, buffer = cv2.imencode('.jpg', processed_img)
            processed_img_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            return jsonify({'error': f'Error encoding processed image: {str(e)}', 'success': False}), 500

        vest_detected = any(d['class'] == 'vest' for d in detections)
        max_confidence = max([d['confidence'] for d in detections if d['class'] == 'vest'], default=0)

        response = {
            'success': True,
            'prediction': 'Vest detected' if vest_detected else 'No vest detected',
            'confidence': max_confidence,
            'detections': detections,
            'processed_image': f'data:image/jpeg;base64,{processed_img_b64}'
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/predict_video_mem', methods=['POST'])
def predict_video_mem():
    try:
        data = request.get_json()
        if not data or 'video' not in data:
            return jsonify({'error': 'No video provided', 'success': False}), 400
        # In-memory processing: decode video to BytesIO
        encoded_video = data['video'].split(',')[1] if ',' in data['video'] else data['video']
        video_bytes = base64.b64decode(encoded_video)
        video_buffer = io.BytesIO(video_bytes)
        
        # Use imageio to read video from in-memory buffer
        reader = imageio.get_reader(video_buffer, 'ffmpeg')
        meta_data = reader.get_meta_data()
        fps = meta_data.get('fps', 25)
        processed_frames = []
        
        # Process each frame with YOLO
        for frame in reader:
            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame, _ = detect_objects(frame_bgr)
            # Convert back to RGB for imageio
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(processed_frame_rgb)
        reader.close()
        
        # Write processed frames to an in-memory video file
        output_buffer = io.BytesIO()
        writer = imageio.get_writer(output_buffer, format='ffmpeg', mode='I', fps=fps)
        for frame in processed_frames:
            writer.append_data(frame)
        writer.close()
        
        processed_video_bytes = output_buffer.getvalue()
        processed_video_b64 = base64.b64encode(processed_video_bytes).decode('utf-8')
        return jsonify({
            'success': True,
            'processed_video': f'data:video/mp4;base64,{processed_video_b64}'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/predict_video_file', methods=['POST'])
def predict_video_file():
    try:
        data = request.get_json()
        if not data or 'video' not in data:
            return jsonify({'error': 'No video provided', 'success': False}), 400
        # Decode video from base64 into BytesIO
        encoded_video = data['video'].split(',')[1] if ',' in data['video'] else data['video']
        video_bytes = base64.b64decode(encoded_video)
        video_buffer = io.BytesIO(video_bytes)
        
        # Use imageio to read the video from the in-memory buffer
        import tempfile
        import imageio

        # Reset the stream position to the beginning
        video_buffer.seek(0)

        # Write the in-memory file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_buffer.read())
            temp_video_path = temp_video.name

        # Now use imageio to read the file from disk
        reader = imageio.get_reader(temp_video_path, 'ffmpeg')

        meta_data = reader.get_meta_data()
        fps = meta_data.get('fps', 25)
        processed_frames = []
        
        # Process each frame using detect_objects
        for frame in reader:
            # imageio returns frames in RGB; convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame, _ = detect_objects(frame_bgr)
            # Convert processed frame back to RGB for imageio output
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(processed_frame_rgb)
        reader.close()
        
        # Save processed video to a file in the static folder
        output_dir = "static/processed_videos"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, f"processed_{int(time.time())}.mp4")
        writer = imageio.get_writer(output_filename, format='ffmpeg', mode='I', fps=fps)
        for frame in processed_frames:
            writer.append_data(frame)
        writer.close()
        
        # Return the URL to the processed video file
        processed_video_url = f"/{output_filename}"
        return jsonify({
            'success': True,
            'processed_video_url': processed_video_url
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(debug=True, host='0.0.0.0', port=port)