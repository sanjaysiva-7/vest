import cv2
import gradio as gr # type: ignore
from ultralytics import YOLO
import numpy as np

# Load trained YOLO model
model = YOLO("test1.pt")  # Replace with your model path

def detect_objects(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(image_rgb)[0]

    # Draw detections on the image
    annotated_frame = results.plot()

    return annotated_frame

# Chatbot logic
def chatbot_response(user_input, chat_history):
    """Simple chatbot for guiding users."""
    user_input = user_input.lower()

    if "image" in user_input:
        bot_response = "📷 Please upload an image, and I will detect reflective jackets for you."
    elif "how" in user_input or "help" in user_input:
        bot_response = "🛠️ Just upload an image, and the AI will detect reflective jackets automatically!"
    elif "hello" in user_input or "hi" in user_input:
        bot_response = "👋 Hello! I can detect reflective jackets in images. Try uploading one."
    else:
        bot_response = "🤖 I'm here to assist you! Just upload an image and see the magic."

    chat_history.append([user_input, bot_response])
    return chat_history

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🦺 Reflective Jacket Detector & AI Chatbot 🚀")
    
    with gr.Row():
        # Detection Section
        with gr.Column():
            iface = gr.Interface(
                fn=detect_objects,
                inputs=gr.Image(type="numpy"),
                outputs=gr.Image(type="numpy"),
                title="Reflective Jacket Detector",
                description="Upload an image, and the model will detect reflective jackets."
            )

        # Chatbot Section
        with gr.Column():
            chatbot = gr.Chatbot(label="🤖 AI Assistant")
            user_input = gr.Textbox(label="Your message")
            send_button = gr.Button("Send")
            send_button.click(chatbot_response, inputs=[user_input, chatbot], outputs=chatbot)

demo.launch()
