import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import os
import shutil

# Load the deepfake detection model
def load_model(model_path=r"C:\Users\bavit\OneDrive\Documents\Desktop\Major Project\DEEPFAKE_DETECTION\deepfake-detector\model\best_deepfake_model.pth"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transform pipeline
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert('RGB')).unsqueeze(0)

# Predict a single image
def predict_image(image_path):
    model = load_model()
    image = Image.open(image_path)
    input_tensor = transform_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return "Fake" if predicted.item() == 1 else "Real"

# Extract frames from a video every `interval` frames
def extract_frames(video_path, output_folder='temp_frames', interval=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_list.append(frame_path)
        count += 1

    cap.release()
    return frame_list

# Predict deepfake likelihood for the entire video using soft voting
def predict_video(video_path, interval=5):
    temp_folder = 'temp_frames'
    frames = extract_frames(video_path, output_folder=temp_folder, interval=interval)
    model = load_model()

    fake_scores = []

    for frame_path in frames:
        try:
            image = Image.open(frame_path)
            input_tensor = transform_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                fake_scores.append(probs[0][1].item())  # Score for 'Fake'
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")

    # Clean up extracted frames
    shutil.rmtree(temp_folder)

    if not fake_scores:
        return "No valid frames found."

    avg_fake_score = sum(fake_scores) / len(fake_scores)
    confidence = avg_fake_score * 100

    result = (
        f"⚠️ Deepfake likely: {confidence:.2f}% confidence.\n"
        if confidence > 50 else
        f"✅ Likely Real: {100 - confidence:.2f}% confidence.\n"
    )
    result += f"Frames analyzed: {len(fake_scores)} | Avg Fake Score: {avg_fake_score:.4f}"
    return result

