import cv2
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO("test1.pt")  # Load your model

# Open video capture
cap = cv2.VideoCapture("abs.mp4")  # Replace with your video file

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    # Run YOLO inference on the GPU (no RGB conversion)
    results = model(frame, device='cuda')[0]  

    # Check if boxes are detected
    if results.boxes is not None and len(results.boxes) > 0:
        # Filter detections based on confidence
        filtered_results = [det for det in results.boxes if det.conf > CONFIDENCE_THRESHOLD]

        if filtered_results:
            first_detection = filtered_results[0]  # Take the first valid detection
            class_name = results.names[int(first_detection.cls)]
            class_count = {class_name: 1}  # Count only the first detected class
            
            print("Detections:", class_count)

            # Keep only the first detection for plotting
            results.boxes = [first_detection]
        else:
            print("No valid detections found.")
            continue  # Skip to the next frame if no valid detections222
    # Use built-in YOLO plotting
    frame = results.plot()  

    # Show output
    cv2.imshow("Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
