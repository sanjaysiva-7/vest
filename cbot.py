import cv2
import gradio as gr # type: ignore
import numpy as np
import tempfile
import os
import traceback
from ultralytics import YOLO

# Load YOLO model
try:
    model = YOLO("test1.pt")  # Update with your trained model path
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for detections

def detect_objects(image):
    """Runs YOLO on an uploaded image and returns the annotated output."""
    try:
        if image is None:
            return "❌ Please upload an image!", None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)[0]
        print(f"📷 Image Detections: {len(results.boxes)} reflective jackets found")

        if len(results.boxes) > 0 and any(results.boxes.conf > CONFIDENCE_THRESHOLD):
            return "✅ Reflective jackets detected!", results.plot()
        else:
            return "❌ No reflective jackets detected.", image

    except Exception as e:
        return f"❌ Error processing image: {str(e)}\n{traceback.format_exc()}", None

def process_video(video_path):
    """Runs YOLO on an uploaded video and returns the processed video."""
    try:
        if not isinstance(video_path, str) or not os.path.exists(video_path):
            return "❌ Please upload a valid video.", None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "❌ Error: Cannot open video file.", None

        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS) / 2))  # Reduce FPS for efficiency

        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = temp_video.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            return "❌ Error: Failed to initialize video writer", None

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 2 == 0:  # Skip every other frame for efficiency
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)[0]
                annotated_frame = results.plot()
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            frame_count += 1

        cap.release()
        out.release()
        return "✅ Video processed successfully!", output_path if os.path.exists(output_path) else None

    except Exception as e:
        return f"❌ Error processing video: {str(e)}\n{traceback.format_exc()}", None

def chatbot_response(user_input, chat_history):
    """Interactive chatbot that guides users."""
    user_input = user_input.lower()

    if "image" in user_input:
        bot_response = "📷 Please upload an image to detect reflective jackets."
    elif "video" in user_input:
        bot_response = "🎥 Please upload a video to process."
    elif "another" in user_input or "again" in user_input:
        bot_response = "🔄 What would you like to process next? Image or video?"
    else:
        bot_response = "🤖 I can detect reflective jackets in images or videos. What would you like to do?"

    chat_history.append([user_input, bot_response])
    return chat_history

# UI setup
with gr.Blocks() as demo:
    gr.Markdown("# 🦺 Reflective Jacket Detection & Chatbot 🚀")
    gr.Markdown("Upload an image or video to detect reflective jackets in the scene.")

    chatbot = gr.Chatbot(label="🤖 Chat with me!")
    user_input = gr.Textbox(label="Your message")
    send_button = gr.Button("Send")

    with gr.Tab("📷 Detect in Image"):
        image_input = gr.Image(type="numpy")
        image_output = gr.Image(type="numpy")
        image_btn = gr.Button("Detect Reflective Jacket")

    with gr.Tab("🎥 Detect in Video"):
        video_input = gr.Video()
        video_output = gr.Video()
        video_btn = gr.Button("Process Video")

    # Chatbot
    send_button.click(chatbot_response, inputs=[user_input, chatbot], outputs=chatbot)

    # Detection
    image_btn.click(detect_objects, inputs=image_input, outputs=[chatbot, image_output])
    video_btn.click(process_video, inputs=video_input, outputs=[chatbot, video_output])

demo.launch(debug=True)
